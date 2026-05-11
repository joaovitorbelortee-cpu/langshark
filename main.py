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

import asyncio
import hmac
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent.checkpointer import CheckpointerProvider, thread_id_for
from agent.graph import build_graph
from agent.qstash import QStashClient
from memory.redis_store import RedisStore


load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("bot-vendas")


# ────────────────────────────────────────────────────────────────────
# Lifespan: cria checkpointer + grafo compartilhados por processo
# ────────────────────────────────────────────────────────────────────

_checkpointer_provider: CheckpointerProvider | None = None
_graph_app: Any | None = None
_worker_task: asyncio.Task[Any] | None = None
_worker_stop_evt: asyncio.Event = asyncio.Event()

import random  # noqa: E402

# Tempo máximo (segundos) que uma mensagem pode ficar na fila antes de descartar.
# Cliente espera 5min sem resposta = melhor descartar que mandar resposta estranha.
QUEUE_MAX_STALE_SECONDS = int(os.getenv("QUEUE_MAX_STALE_SECONDS", "300"))
# Intervalo de poll quando queue vazia (Upstash REST não suporta BRPOP).
QUEUE_POLL_INTERVAL_S = float(os.getenv("QUEUE_POLL_INTERVAL_S", "0.5"))

# ────── Inter-lead delay (anti-spam humano) ──────
# Pausa randomizada entre processar leads DIFERENTES. Mesmo lead em rajada não
# sofre delay (continuação natural). Range varia c/ carga da queue:
#   carga baixa (≤2)   → 1–3 min  (cliente espera pouco)
#   carga média (3–5)  → 1–4 min  (default)
#   carga alta (>5)    → 2–5 min  (humano "ocupado" demora mais)
INTER_LEAD_DELAY_ENABLED = os.getenv("INTER_LEAD_DELAY_ENABLED", "1") == "1"
INTER_LEAD_LOW_THRESHOLD = int(os.getenv("INTER_LEAD_LOW_THRESHOLD", "2"))
INTER_LEAD_HIGH_THRESHOLD = int(os.getenv("INTER_LEAD_HIGH_THRESHOLD", "5"))
INTER_LEAD_LOW_RANGE_S = (60, 180)     # 1-3 min
INTER_LEAD_NORMAL_RANGE_S = (60, 240)  # 1-4 min
INTER_LEAD_HIGH_RANGE_S = (120, 300)   # 2-5 min


def _calc_inter_lead_delay(qsize: int) -> float:
    """
    Retorna sleep em segundos antes de processar próximo lead diferente.
    Faixa adapta à carga: queue grande = humano "ocupado" demora mais.
    """
    if qsize <= INTER_LEAD_LOW_THRESHOLD:
        lo, hi = INTER_LEAD_LOW_RANGE_S
    elif qsize > INTER_LEAD_HIGH_THRESHOLD:
        lo, hi = INTER_LEAD_HIGH_RANGE_S
    else:
        lo, hi = INTER_LEAD_NORMAL_RANGE_S
    return random.uniform(lo, hi)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _checkpointer_provider, _graph_app, _worker_task
    _checkpointer_provider = CheckpointerProvider()
    cp = await _checkpointer_provider.shared()
    _graph_app = build_graph(checkpointer=cp)
    log.info("[startup] grafo compilado (checkpointer=%s)", _checkpointer_provider.kind)

    # Sobe worker FIFO — drena queue Redis serialmente, 1 mensagem por vez,
    # anti-ban WhatsApp (paralelismo dispararia N envios simultâneos).
    _worker_stop_evt.clear()
    _worker_task = asyncio.create_task(_queue_worker_loop(), name="queue-worker")
    log.info("[startup] FIFO queue worker iniciado")

    try:
        yield
    finally:
        _worker_stop_evt.set()
        if _worker_task and not _worker_task.done():
            _worker_task.cancel()
            try:
                await _worker_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if _checkpointer_provider:
            await _checkpointer_provider.aclose()
        _graph_app = None


limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(title="bot-vendas", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


import secrets  # noqa: E402


CSRF_COOKIE = "csrftoken"
CSRF_HEADER = "x-csrf-token"
_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Webhooks externos (Evolution, QStash) usam auth via secret e NÃO devem ter CSRF.
_CSRF_EXEMPT_PATHS = ("/webhook", "/api/trigger-followup", "/health")


def _is_csrf_exempt(path: str) -> bool:
    return any(path.startswith(p) for p in _CSRF_EXEMPT_PATHS)


@app.middleware("http")
async def _security_and_csrf(request: Request, call_next):
    """
    Combina security headers + CSRF double-submit cookie.

    CSRF: SameSite=strict no cookie session já bloqueia maioria. Token double-submit
    é defesa secundária pra defesa-em-profundidade contra CSRF + bug em browser legado.
    """
    path = request.url.path
    method = request.method.upper()

    # CSRF check em métodos mutantes (POST/PATCH/PUT/DELETE) — exceto webhooks externos.
    if method not in _CSRF_SAFE_METHODS and not _is_csrf_exempt(path):
        cookie_token = request.cookies.get(CSRF_COOKIE, "")
        header_token = request.headers.get(CSRF_HEADER, "")
        # Login submit (form) usa SameSite=strict como única defesa — exempta primeira chamada.
        if path == "/admin/login" and method == "POST":
            pass  # form submit, sem token ainda
        elif not cookie_token or not header_token or not hmac.compare_digest(cookie_token, header_token):
            return _json_response(403, {"detail": "CSRF token missing or invalid"})

    response = await call_next(request)

    # Emite/renova cookie CSRF em qualquer GET de página HTML (não em JSON APIs).
    if method in _CSRF_SAFE_METHODS and (path.startswith("/admin") and not path.startswith("/admin/static") and not path.startswith("/api/admin")):
        if not request.cookies.get(CSRF_COOKIE):
            response.set_cookie(
                key=CSRF_COOKIE,
                value=secrets.token_urlsafe(32),
                max_age=24 * 3600,
                httponly=False,  # double-submit precisa ser lido por JS
                secure=True,
                samesite="strict",
                path="/",
            )

    # Security headers
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    if path.startswith("/admin") and not path.startswith("/admin/static"):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'",
        )
    return response


def _json_response(status: int, body: dict) -> Any:
    """Helper minimal pra resposta JSON dentro de middleware (evita import cíclico)."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content=body)

# Painel admin
from pathlib import Path  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from panel.api import admin_router  # noqa: E402
from panel.views import views_router  # noqa: E402

app.include_router(admin_router, prefix="/api/admin")
app.include_router(views_router, prefix="/admin")
app.mount(
    "/admin/static",
    StaticFiles(directory=str(Path(__file__).parent / "panel" / "static")),
    name="admin-static",
)

redis = RedisStore()
qstash = QStashClient()


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
@limiter.limit("60/minute")
async def webhook(request: Request) -> dict:
    """
    Recebe webhook Evolution, valida, enfileira na queue Redis e retorna 200 imediato.

    FIFO Worker (em _queue_worker_loop) drena queue 1 mensagem por vez —
    anti-ban WhatsApp (paralelismo dispararia N envios simultâneos do mesmo
    número e WhatsApp suspende a conta).
    """
    req = request  # slowapi exige 'request' nomeado
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

    # Sinaliza ao QStash KillSwitch — lead enviou msg, cancela follow-up pendente
    try:
        await redis.set_last_from(instance, phone, "lead")
    except Exception:  # noqa: BLE001
        pass

    # Empilha na FIFO global — worker drena
    queue_payload = {
        "kind": "inbound",
        "instance": instance,
        "phone": phone,
        "message_id": message_id,
        "enqueued_at": time.time(),
        "initial_state": {
            "project_id": project_id_hint,
            "instance_name": instance,
            "phone": phone,
            "push_name": push_name,
            "user_message": user_message,
            "media_mime": media["mime"] if media else None,
            "media_base64": media["base64"] if media else None,
            "message_id": message_id,
            "messages": [],
        },
    }
    qsize = await redis.enqueue_message(queue_payload)
    log.info("[queue] enqueued inbound %s/%s (size=%d)", instance, phone, qsize)
    return {"ok": True, "queued": True, "queue_size": qsize}


# ────────────────────────────────────────────────────────────────────
# Follow-up trigger (callback agendado pelo QStash)
# ────────────────────────────────────────────────────────────────────

@app.post("/api/trigger-followup")
@limiter.limit("30/minute")
async def trigger_followup(request: Request) -> dict:
    """
    Endpoint chamado pelo QStash após N minutos.
    Body: {project_id, instance_name, phone, push_name}.

    KillSwitch: se o lead respondeu nesse meio-tempo (last_message_from=lead),
    cancela. Caso contrário, enfileira follow-up na FIFO queue.
    """
    req = request
    _check_auth(req)
    body = await req.json()

    project_id = body.get("project_id") or "padrao"
    instance = body.get("instance_name") or ""
    phone = body.get("phone") or ""
    push_name = body.get("push_name") or ""

    if not instance or not phone:
        raise HTTPException(status_code=400, detail="instance_name and phone required")

    # KillSwitch — checa última atividade no Redis
    last_from = await redis._cmd("GET", f"last_from:{instance}:{phone}")
    if last_from == "lead":
        log.info("[followup] killswitch %s/%s — lead respondeu", instance, phone)
        return {"ok": True, "skipped": "killswitch_lead_replied"}

    # Enfileira follow-up (FIFO global garante anti-spam ao misturar com inbounds)
    queue_payload = {
        "kind": "followup",
        "instance": instance,
        "phone": phone,
        "message_id": "",
        "enqueued_at": time.time(),
        "initial_state": {
            "project_id": project_id,
            "instance_name": instance,
            "phone": phone,
            "push_name": push_name,
            "user_message": "",
            "media_mime": None,
            "media_base64": None,
            "message_id": "",
            "intent": "follow_up",
            "messages": [],
        },
    }
    qsize = await redis.enqueue_message(queue_payload)
    log.info("[queue] enqueued followup %s/%s (size=%d)", instance, phone, qsize)
    return {"ok": True, "queued": True, "queue_size": qsize}


# ────────────────────────────────────────────────────────────────────
# FIFO worker — drena queue 1 mensagem por vez (anti-ban WhatsApp)
# ────────────────────────────────────────────────────────────────────

async def _queue_worker_loop() -> None:
    """
    Loop infinito até _worker_stop_evt. RPOP da queue, processa, repete.
    Aplica delay aleatório entre leads DIFERENTES (anti-spam humano).
    Pausa QUEUE_POLL_INTERVAL_S quando queue vazia (Upstash REST não suporta BRPOP).
    """
    backoff = QUEUE_POLL_INTERVAL_S
    last_processed_phone: str = ""
    while not _worker_stop_evt.is_set():
        try:
            payload = await redis.dequeue_message()
            if payload is None:
                await asyncio.sleep(backoff)
                backoff = min(2.0, backoff * 1.2)
                continue
            backoff = QUEUE_POLL_INTERVAL_S

            # Descarta msgs muito antigas (cliente já desistiu de esperar)
            enq_at = float(payload.get("enqueued_at") or 0)
            if enq_at and (time.time() - enq_at) > QUEUE_MAX_STALE_SECONDS:
                log.warning(
                    "[queue] descartando stale (%.1fs) kind=%s %s/%s",
                    time.time() - enq_at,
                    payload.get("kind"),
                    payload.get("instance"),
                    payload.get("phone"),
                )
                continue

            current_phone = payload.get("phone") or ""

            # Inter-lead delay: só aplica quando trocamos de lead.
            # Mesmo lead em rajada não pausa (continuação natural da conversa).
            if (
                INTER_LEAD_DELAY_ENABLED
                and last_processed_phone
                and current_phone
                and current_phone != last_processed_phone
            ):
                qsize_now = await redis.queue_length()
                delay = _calc_inter_lead_delay(qsize_now)
                log.info(
                    "[queue] inter-lead delay %.1fs (qsize=%d, %s→%s)",
                    delay, qsize_now, last_processed_phone, current_phone,
                )
                # Sleep interruptível pelo stop_evt — encerramento limpo
                try:
                    await asyncio.wait_for(_worker_stop_evt.wait(), timeout=delay)
                    # Se chegou aqui, stop_evt setado durante o sleep → sair
                    return
                except asyncio.TimeoutError:
                    pass  # delay completo, segue

            try:
                await _process_queued(payload)
            finally:
                # Atualiza last_processed_phone MESMO em exceção — evita
                # delay duplicado se mesma rajada de erro continua.
                last_processed_phone = current_phone or last_processed_phone
        except asyncio.CancelledError:
            log.info("[queue] worker cancelado")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("[queue] erro no worker: %s", exc)
            await asyncio.sleep(1.0)


async def _process_queued(payload: dict[str, Any]) -> None:
    """Processa 1 mensagem da queue: lock per phone + run graph + schedule follow-up."""
    instance = payload.get("instance") or ""
    phone = payload.get("phone") or ""
    kind = payload.get("kind") or "inbound"
    initial_state = payload.get("initial_state") or {}
    project_id_hint = initial_state.get("project_id") or "padrao"

    if not instance or not phone:
        log.warning("[queue] payload sem instance/phone, descartando: %s", payload)
        return

    # Lock per (instance, phone) — protege contra concorrência caso worker
    # restart deixe entry duplicada.
    got_lock = await redis.acquire_lock(instance, phone, ttl_seconds=120)
    if not got_lock:
        # Lock ainda detido — re-enqueue no fim, processa outro lead enquanto isso.
        log.info("[queue] lock_held %s/%s — re-enqueue", instance, phone)
        await redis.requeue_head(payload)
        await asyncio.sleep(1.0)
        return

    try:
        thread_id = thread_id_for(project_id_hint, instance, phone)
        final_state = await _run_graph_streaming(initial_state, thread_id)
        log.info(
            "[queue] processed %s %s/%s intent=%s sent=%s",
            kind, instance, phone,
            final_state.get("intent"),
            final_state.get("sent_count", 0),
        )

        # Marca quem foi o último a falar — KillSwitch usa isso pra cancelar follow-ups
        try:
            await redis.set_last_from(instance, phone, "agent")
        except Exception:  # noqa: BLE001
            pass

        # Se este turno foi follow-up disparado pelo bot, incrementa contador
        # (próxima decisão do strategist saberá quantas tentativas já houve).
        attempts_after_incr = 0
        if kind == "followup":
            try:
                attempts_after_incr = await redis.increment_followup_attempts(instance, phone)
                log.info("[strategist] %s/%s attempts=%d", instance, phone, attempts_after_incr)
            except Exception:  # noqa: BLE001
                pass

        # Schedule próximo follow-up se IA/strategist pediu E lead não converteu
        schedule_min = final_state.get("schedule_minutes")
        has_converted = final_state.get("has_converted", False)
        strategy = final_state.get("follow_up_strategy") or {}
        killswitch = bool(strategy.get("killswitch_permanent"))

        # Toggle per-projeto: follow_up_enabled=False desativa follow-ups
        # sem desativar o bot inteiro (is_active continua separado).
        try:
            from panel.cache import get_project_config_cache
            project_cfg = await get_project_config_cache().get(
                final_state.get("project_id") or project_id_hint
            )
            followup_enabled = project_cfg.get("followup_enabled")
            if followup_enabled is False:
                schedule_min = None
                log.info("[strategist] follow_up_enabled=false → skip schedule %s/%s",
                         instance, phone)
        except Exception:  # noqa: BLE001
            pass

        # HARD GATE: N+ tentativas sem resposta → mata follow-up permanentemente
        # Strategist agora roda também em follow_up turns, mas este gate é
        # defesa em profundidade contra LLM/strategist falharem.
        FOLLOWUP_MAX_ATTEMPTS = int(os.getenv("FOLLOWUP_MAX_ATTEMPTS", "10"))
        if kind == "followup" and attempts_after_incr >= FOLLOWUP_MAX_ATTEMPTS:
            killswitch = True
            schedule_min = None
            log.info(
                "[strategist] hard cap atingido %s/%s (attempts=%d) — marca LOST",
                instance, phone, attempts_after_incr,
            )
            # Atualiza snapshot do lead pro painel ver killswitch
            try:
                from datetime import datetime as _dt, timezone as _tz
                snapshot = await redis.get_lead_status(instance, phone) or {}
                snapshot.update({
                    "instance": instance,
                    "phone": phone,
                    "killswitch_permanent": True,
                    "temperatura": "STOP",
                    "razao": f"Lost — {FOLLOWUP_MAX_ATTEMPTS}+ tentativas sem resposta",
                    "attempts_made": attempts_after_incr,
                    "last_decision_at": _dt.now(_tz.utc).isoformat(),
                    "next_followup_at": None,
                    "agendar_minutos": 0,
                })
                await redis.set_lead_status(instance, phone, snapshot)
            except Exception:  # noqa: BLE001
                pass

        if schedule_min and not has_converted and not killswitch and qstash.enabled:
            r = await qstash.schedule_followup(
                project_id=final_state.get("project_id") or project_id_hint,
                instance=instance,
                phone=phone,
                delay_minutes=schedule_min,
                push_name=initial_state.get("push_name", ""),
            )
            log.info(
                "[qstash] schedule_followup temp=%s abord=%s min=%d → %s",
                strategy.get("temperatura", "?"),
                strategy.get("abordagem", "?"),
                schedule_min, r,
            )
        elif killswitch:
            log.info("[strategist] killswitch ativo %s/%s — não agenda follow-up", instance, phone)
    finally:
        await redis.release_lock(instance, phone)


@app.get("/health")
async def health() -> dict:
    qsize = 0
    try:
        qsize = await redis.queue_length()
    except Exception:  # noqa: BLE001
        pass
    worker_alive = bool(_worker_task and not _worker_task.done())
    return {
        "ok": True,
        "redis": "remote" if redis.remote_enabled else "local-fallback",
        "evolution": bool(os.getenv("EVOLUTION_API_URL")),
        "checkpointer": _checkpointer_provider.kind if _checkpointer_provider else "uninit",
        "qstash": qstash.enabled,
        "queue_size": qsize,
        "worker_alive": worker_alive,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD")),
    )
