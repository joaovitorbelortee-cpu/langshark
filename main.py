"""
Webhook FastAPI compatível com a Evolution API (mesma rota do bot antigo).

Eventos tratados:
  - messages.upsert   → roda o grafo de vendas
  - presence.update   → marca digitando no Redis (informativo)
  - connection.update → log apenas
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from agent.graph import get_graph
from agent.tools import (
    EvolutionClient,
    jitter_between_bubbles_ms,
    typing_delay_ms,
)
from memory.redis_store import RedisStore


load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("bot-vendas")


app = FastAPI(title="bot-vendas")

evolution = EvolutionClient()
redis = RedisStore()


# ────────────────────────────────────────────────────────────────────
# Helpers de auth (timing-safe igual ao bot antigo)
# ────────────────────────────────────────────────────────────────────

def _safe_eq(a: str, b: str) -> bool:
    if not a or not b or len(a) != len(b):
        return False
    return hmac.compare_digest(a, b)


def _check_auth(req: Request) -> None:
    secret = (os.getenv("WEBHOOK_SECRET") or "").strip()
    evo_key = (os.getenv("EVOLUTION_API_KEY") or "").strip()
    if not secret and not evo_key:
        raise HTTPException(status_code=503, detail="Server misconfiguration: no webhook secret")

    incoming = (
        req.headers.get("apikey")
        or req.headers.get("x-webhook-token")
        or req.query_params.get("token")
        or req.headers.get("global-api-key")
        or ""
    ).strip()

    if not (_safe_eq(incoming, secret) or _safe_eq(incoming, evo_key)):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ────────────────────────────────────────────────────────────────────
# Resolver project_id pra instância (multi-tenant)
# ────────────────────────────────────────────────────────────────────

async def _resolve_project_id(instance: str, query_project: str | None) -> str:
    """
    No bot antigo, vinha de instance_projects (Supabase).
    Aqui aceitamos via query (?project_id=) ou env padrão. Default: "padrao".
    """
    if query_project:
        return query_project
    return os.getenv("DEFAULT_PROJECT_ID", "padrao")


# ────────────────────────────────────────────────────────────────────
# Extração de payload Evolution (mesmas regras do webhook antigo)
# ────────────────────────────────────────────────────────────────────

_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MAX_TEXT = 4096


def _strip_ctrl(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return _CTRL_RE.sub("", s)[:MAX_TEXT]


def _extract_text(message: dict) -> str:
    if message.get("conversation"):
        return _strip_ctrl(message["conversation"])
    if message.get("extendedTextMessage", {}).get("text"):
        return _strip_ctrl(message["extendedTextMessage"]["text"])
    for k in ("imageMessage", "videoMessage", "documentMessage"):
        cap = message.get(k, {}).get("caption")
        if cap:
            return _strip_ctrl(cap)
    return ""


def _extract_media(message: dict, body_data: dict) -> dict | None:
    for k, default_mime in (
        ("imageMessage", "image/jpeg"),
        ("audioMessage", "audio/ogg"),
        ("videoMessage", "video/mp4"),
        ("documentMessage", "application/octet-stream"),
    ):
        if k in message:
            blk = message[k]
            return {
                "mime": blk.get("mimetype", default_mime),
                "caption": blk.get("caption", ""),
                "base64": body_data.get("message", {}).get("base64") or message.get("base64"),
            }
    return None


# ────────────────────────────────────────────────────────────────────
# Envio de bolhas (replica simulateTyping + send_text + jitter)
# ────────────────────────────────────────────────────────────────────

async def _send_chunks(instance: str, phone: str, chunks: list[str]) -> None:
    for i, chunk in enumerate(chunks):
        delay_ms = typing_delay_ms(chunk)
        await evolution.send_typing(instance, phone, duration_ms=delay_ms)
        await asyncio.sleep(delay_ms / 1000)
        await evolution.send_text(instance, phone, chunk)
        if i < len(chunks) - 1:
            await asyncio.sleep(jitter_between_bubbles_ms() / 1000)


# ────────────────────────────────────────────────────────────────────
# Endpoint principal
# ────────────────────────────────────────────────────────────────────

@app.post("/webhook/evolution")
@app.post("/webhook")
async def webhook(req: Request) -> dict:
    _check_auth(req)
    body = await req.json()

    event = (body.get("event") or "").replace("whatsapp.", "").lower()
    instance = (
        req.query_params.get("instance_name")
        or body.get("instance")
        or body.get("instanceName")
        or os.getenv("EVOLUTION_INSTANCE", "botzap")
    )

    if event == "connection.update":
        state = (body.get("data") or {}).get("state") or body.get("state")
        log.info("[conn] %s state=%s", instance, state)
        return {"ok": True, "connectionState": state}

    if event == "presence.update":
        # Não bloqueia o fluxo principal; só registra no Redis (informativo).
        return {"ok": True, "skipped": "presence"}

    if event != "messages.upsert":
        return {"ok": True, "skipped": event}

    data = body.get("data") or {}
    key = data.get("key") or {}
    message = data.get("message") or {}

    if key.get("fromMe"):
        return {"ok": True, "skipped": "fromMe"}

    remote_jid: str = key.get("remoteJid") or ""
    if remote_jid.endswith("@g.us") or "@broadcast" in remote_jid:
        return {"ok": True, "skipped": "group/broadcast"}

    phone = remote_jid.replace("@s.whatsapp.net", "").replace("@g.us", "")
    if not re.fullmatch(r"\d{10,15}", phone or ""):
        return {"ok": True, "skipped": "invalid phone"}

    text = _extract_text(message)
    media = _extract_media(message, data)
    if not text and not media:
        return {"ok": True, "skipped": "no content"}

    message_id = key.get("id") or ""
    if message_id:
        first = await redis.mark_message_processed(instance, phone, message_id)
        if not first:
            return {"ok": True, "skipped": "duplicate", "messageId": message_id}

    project_id = await _resolve_project_id(instance, req.query_params.get("project_id"))
    push_name = data.get("pushName") or ""

    user_message = text or (media.get("caption") if media else "") or "[mídia]"
    log.info("[msg] %s/%s/%s: %s", project_id, instance, phone, user_message[:80])

    # Roda o grafo. As mensagens chegam sequencialmente — se quiser fila,
    # plugue aqui o agendador (QStash/Celery/etc).
    graph = get_graph()
    initial_state = {
        "project_id": project_id,
        "instance_name": instance,
        "phone": phone,
        "push_name": push_name,
        "user_message": user_message,
        "media_mime": media["mime"] if media else None,
        "media_base64": media["base64"] if media else None,
        "messages": [],
    }
    final_state = await graph.ainvoke(initial_state)

    chunks: list[str] = final_state.get("chunks") or []
    if not chunks:
        log.warning("[graph] sem resposta gerada para %s", phone)
        return {"ok": True, "processed": True, "empty_reply": True}

    # Reaction (se a IA pediu) — antes das bolhas, igual ao bot antigo.
    if final_state.get("react_emoji") and message_id:
        try:
            await evolution.send_reaction(instance, phone, message_id, final_state["react_emoji"])
        except Exception as exc:
            log.warning("[react] falha: %s", exc)

    await _send_chunks(instance, phone, chunks)

    # TODO opcional: agendar follow-up via QStash usando final_state["schedule_minutes"]
    # quando has_converted=False. Mantido como ponto de extensão para não trazer
    # o ecossistema QStash inteiro pra cá agora.

    return {
        "ok": True,
        "processed": True,
        "intent": final_state.get("intent"),
        "has_converted": final_state.get("has_converted", False),
        "schedule_minutes": final_state.get("schedule_minutes"),
        "bubbles": len(chunks),
    }


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "redis": "remote" if redis.remote_enabled else "local-fallback",
        "evolution": bool(os.getenv("EVOLUTION_API_URL")),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD")),
    )
