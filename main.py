"""
Webhook FastAPI compatível com a Evolution API (mesma rota do bot antigo).

Responsabilidades do main.py (slim):
  1. Auth timing-safe do webhook
  2. Extração de campos do payload Evolution
  3. Dedup por messageId no Redis
  4. Dispara o grafo LangGraph com `astream_events` (logs por nó)
  5. Lifecycle do checkpointer (open/close)

TUDO o que é decisão, RAG, envio, persistência → vive dentro do grafo.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from agent.checkpointer import CheckpointerProvider, thread_id_for
from agent.graph import build_graph
from memory.redis_store import RedisStore


load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("bot-vendas")


# ────────────────────────────────────────────────────────────────────
# Lifespan: cria checkpointer + grafo compartilhados por processo
# ────────────────────────────────────────────────────────────────────

_checkpointer_provider: CheckpointerProvider | None = None
_graph_app: Any | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _checkpointer_provider, _graph_app
    _checkpointer_provider = CheckpointerProvider()
    cp = await _checkpointer_provider.shared()
    _graph_app = build_graph(checkpointer=cp)
    log.info("[startup] grafo compilado (checkpointer=%s)", _checkpointer_provider.kind)
    try:
        yield
    finally:
        if _checkpointer_provider:
            await _checkpointer_provider.aclose()
        _graph_app = None


app = FastAPI(title="bot-vendas", lifespan=lifespan)
redis = RedisStore()


# ────────────────────────────────────────────────────────────────────
# Auth (timing-safe igual ao bot antigo)
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
# Extração de payload Evolution
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
# Streaming do grafo com logs por nó
# ────────────────────────────────────────────────────────────────────

_TRACED_NODES = {
    "tenant_resolver", "load_history", "summarize", "vision",
    "detect_intent", "retrieve_for_close", "retrieve_for_respond",
    "close_sale", "respond", "greeting", "objection", "follow_up",
    "flow_executor", "persist", "send",
}


async def _run_graph_streaming(initial_state: dict, thread_id: str) -> dict:
    """
    Executa o grafo com astream_events, logando cada nó conforme entra/sai.
    Retorna o final_state quando o stream termina.
    """
    assert _graph_app is not None, "graph not initialized"
    cfg = {"configurable": {"thread_id": thread_id}}
    final_state: dict[str, Any] = {}

    async for ev in _graph_app.astream_events(initial_state, config=cfg, version="v2"):
        kind = ev.get("event") or ""
        name = ev.get("name") or ""
        if name not in _TRACED_NODES:
            continue
        if kind == "on_chain_start":
            log.info("[graph] ▶ %s", name)
        elif kind == "on_chain_end":
            output = (ev.get("data") or {}).get("output") or {}
            keys = ",".join(sorted(output.keys())) if isinstance(output, dict) else "—"
            log.info("[graph] ✓ %s (patch=%s)", name, keys)

    # Recupera o state final pelo checkpoint.
    snap = await _graph_app.aget_state(cfg)
    if snap:
        final_state = dict(snap.values or {})
    return final_state


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

    project_id_hint = req.query_params.get("project_id") or ""
    push_name = data.get("pushName") or ""
    user_message = text or (media.get("caption") if media else "") or "[mídia]"
    log.info("[msg] %s/%s/%s: %s", project_id_hint or "?", instance, phone, user_message[:80])

    initial_state = {
        "project_id": project_id_hint,
        "instance_name": instance,
        "phone": phone,
        "push_name": push_name,
        "user_message": user_message,
        "media_mime": media["mime"] if media else None,
        "media_base64": media["base64"] if media else None,
        "message_id": message_id,
        "messages": [],
    }

    thread_id = thread_id_for(
        project_id=project_id_hint or "padrao",
        instance=instance,
        phone=phone,
    )
    final_state = await _run_graph_streaming(initial_state, thread_id)

    return {
        "ok": True,
        "processed": True,
        "intent": final_state.get("intent"),
        "has_converted": final_state.get("has_converted", False),
        "schedule_minutes": final_state.get("schedule_minutes"),
        "sent_count": final_state.get("sent_count", 0),
        "flow_dispatched": final_state.get("flow_dispatched", False),
    }


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "redis": "remote" if redis.remote_enabled else "local-fallback",
        "evolution": bool(os.getenv("EVOLUTION_API_URL")),
        "checkpointer": _checkpointer_provider.kind if _checkpointer_provider else "uninit",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD")),
    )
