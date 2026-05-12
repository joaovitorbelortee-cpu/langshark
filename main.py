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
_store_provider: Any | None = None  # StoreProvider, lazy import pra evitar erro se módulo faltar
_graph_app: Any | None = None
_worker_task: asyncio.Task[Any] | None = None
_inbox_drainer_task: asyncio.Task[Any] | None = None
_worker_stop_evt: asyncio.Event = asyncio.Event()

import random  # noqa: E402

# Tempo máximo (segundos) que uma mensagem pode ficar na fila antes de descartar.
# Cliente espera 5min sem resposta = melhor descartar que mandar resposta estranha.
QUEUE_MAX_STALE_SECONDS = int(os.getenv("QUEUE_MAX_STALE_SECONDS", "300"))
# Intervalo de poll quando queue vazia (Upstash REST não suporta BRPOP).
QUEUE_POLL_INTERVAL_S = float(os.getenv("QUEUE_POLL_INTERVAL_S", "0.5"))

# ────── Smart Inter-message Delay (anti-ban gaussian) ──────
# Pausa entre msgs do bot (não entre leads específicos — bot revezando).
# Research-based (baileys-antiban + green-api + chatarmin 2025-2026):
#   - Mínimo 45s entre msgs pra leads diferentes (NUNCA furar)
#   - Gaussian jitter > uniform (padrão temporal humano)
#   - Adaptativo por carga (queue size):
#     calmo (0-2)   → mean=110s std=25s [75, 150]
#     normal (3-5)  → mean=85s  std=20s [60, 120]
#     pico (6+)     → mean=65s  std=15s [45, 90]
#   - HOT lane (lead fechando): mean=60s std=12s [45, 75]
INTER_LEAD_DELAY_ENABLED = os.getenv("INTER_LEAD_DELAY_ENABLED", "1") == "1"

# Threshold de carga (inclui inbox + queue)
SCHED_LOW_THRESHOLD = int(os.getenv("SCHED_LOW_THRESHOLD", "2"))
SCHED_HIGH_THRESHOLD = int(os.getenv("SCHED_HIGH_THRESHOLD", "5"))

# Gaussian: (mean, stddev, hard_min, hard_max). Min sempre >= 45s research-based.
SCHED_DELAY_CALM = (110.0, 25.0, 75.0, 150.0)
SCHED_DELAY_NORMAL = (85.0, 20.0, 60.0, 120.0)
SCHED_DELAY_PEAK = (65.0, 15.0, 45.0, 90.0)
SCHED_DELAY_HOT = (60.0, 12.0, 45.0, 75.0)

# Estágios que entram em HOT lane (lead quente, prioridade)
HOT_STAGES = {"preco", "fechamento"}


def _gauss_delay(profile: tuple[float, float, float, float]) -> float:
    """Sample delay gaussian clamped pra [hard_min, hard_max]. Anti-ban research-based."""
    mean, std, lo, hi = profile
    v = random.gauss(mean, std)
    return max(lo, min(hi, v))


def _calc_inter_lead_delay(qsize: int, hot: bool = False) -> float:
    """
    Delay entre msgs do bot. Adaptativo por carga + HOT lane.
    HOT lane (lead fechando) sempre tem mean menor pra preservar emoção.
    """
    if hot:
        return _gauss_delay(SCHED_DELAY_HOT)
    if qsize <= SCHED_LOW_THRESHOLD:
        return _gauss_delay(SCHED_DELAY_CALM)
    if qsize > SCHED_HIGH_THRESHOLD:
        return _gauss_delay(SCHED_DELAY_PEAK)
    return _gauss_delay(SCHED_DELAY_NORMAL)


async def _bootstrap_evolution_webhooks() -> None:
    """
    Auto-config Evolution webhooks no startup. Idempotente — só atualiza se
    URL diferente do expected ou webhook disabled. Failure soft (não crash
    app se Evolution unreachable).

    Resolve problema: Evolution criada antes do app ter PUBLIC_BASE_URL =
    webhook URL stale → bot não recebe msgs. Em vez de pedir pro user rodar
    curl manual, app fixa sozinho no startup.
    """
    base_url = (os.getenv("EVOLUTION_API_URL") or "").rstrip("/")
    api_key = (os.getenv("EVOLUTION_API_KEY") or "").strip()
    secret = (os.getenv("WEBHOOK_SECRET") or "").strip()
    pub_raw = (os.getenv("PUBLIC_BASE_URL") or os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip()

    if not all([base_url, api_key, secret, pub_raw]):
        log.warning(
            "[bootstrap] evolution-webhook skip — missing env (api_url=%s api_key=%s secret=%s pub=%s)",
            bool(base_url), bool(api_key), bool(secret), bool(pub_raw),
        )
        return

    if not pub_raw.startswith("http"):
        pub_raw = "https://" + pub_raw
    expected = f"{pub_raw.rstrip('/')}/webhook/evolution"

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{base_url}/instance/fetchInstances", headers={"apikey": api_key})
            r.raise_for_status()
            instances = r.json() or []
            if not isinstance(instances, list):
                instances = []

            for item in instances:
                inst = item.get("instance") if isinstance(item, dict) and "instance" in item else item
                if not isinstance(inst, dict):
                    continue
                name = inst.get("instanceName") or inst.get("name") or ""
                if not name:
                    continue

                # Verifica webhook atual
                needs_fix = True
                try:
                    wr = await c.get(f"{base_url}/webhook/find/{name}", headers={"apikey": api_key})
                    if wr.is_success and wr.content:
                        wd = wr.json() or {}
                        actual_url = (wd.get("url") or "").rstrip("/")
                        enabled = bool(wd.get("enabled"))
                        if enabled and actual_url == expected.rstrip("/"):
                            needs_fix = False
                except httpx.HTTPError:
                    pass  # assume needs_fix

                if not needs_fix:
                    log.info("[bootstrap] webhook %s OK (%s)", name, expected)
                    continue

                # Auto-fix
                try:
                    fr = await c.post(
                        f"{base_url}/webhook/set/{name}",
                        headers={"apikey": api_key, "Content-Type": "application/json"},
                        json={
                            "webhook": {
                                "url": expected,
                                "enabled": True,
                                "webhookByEvents": False,
                                "webhookBase64": True,
                                "headers": {"apikey": secret, "Content-Type": "application/json"},
                                "events": ["MESSAGES_UPSERT", "MESSAGES_UPDATE", "CONNECTION_UPDATE", "SEND_MESSAGE"],
                            }
                        },
                    )
                    if fr.is_success:
                        log.info("[bootstrap] webhook %s FIXED → %s", name, expected)
                    else:
                        log.warning(
                            "[bootstrap] webhook %s fix failed (%d): %s",
                            name, fr.status_code, fr.text[:200],
                        )
                except httpx.HTTPError as e:
                    log.warning("[bootstrap] webhook %s fix erro: %s", name, e)
    except httpx.HTTPError as e:
        log.warning("[bootstrap] Evolution unreachable (%s) — skip webhook bootstrap", e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _checkpointer_provider, _store_provider, _graph_app, _worker_task
    _checkpointer_provider = CheckpointerProvider()
    cp = await _checkpointer_provider.shared()

    # Store opcional: cross-thread long-term memory (lead_facts, prefs).
    # Mesmo Redis URL do checkpointer. Se falhar, fallback InMemoryStore.
    store = None
    try:
        from agent.store import StoreProvider, set_shared_store
        _store_provider = StoreProvider()
        store = await _store_provider.shared()
        set_shared_store(store)  # nós acessam via get_shared_store()
        log.info("[startup] store inicializado (%s)", _store_provider.kind)
    except Exception as exc:  # noqa: BLE001
        log.warning("[startup] store falhou (%s) — graph sem store", exc)

    _graph_app = build_graph(checkpointer=cp, store=store)
    log.info("[startup] grafo compilado (checkpointer=%s, store=%s)",
             _checkpointer_provider.kind,
             _store_provider.kind if _store_provider else "none")

    # Sobe worker FIFO — drena queue Redis serialmente, 1 mensagem por vez,
    # anti-ban WhatsApp (paralelismo dispararia N envios simultâneos).
    _worker_stop_evt.clear()
    global _inbox_drainer_task
    _worker_task = asyncio.create_task(_queue_worker_loop(), name="queue-worker")
    _inbox_drainer_task = asyncio.create_task(_inbox_drainer_loop(), name="inbox-drainer")
    log.info("[startup] FIFO worker + inbox drainer iniciados (debounce=%.1fs)",
             INBOX_DEBOUNCE_S)

    # Auto-fix Evolution webhooks (background — não bloqueia startup)
    asyncio.create_task(_bootstrap_evolution_webhooks(), name="bootstrap-webhooks")

    try:
        yield
    finally:
        _worker_stop_evt.set()
        for t, name in ((_worker_task, "queue-worker"), (_inbox_drainer_task, "inbox-drainer")):
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        if _checkpointer_provider:
            await _checkpointer_provider.aclose()
        if _store_provider:
            try:
                from agent.store import set_shared_store
                set_shared_store(None)
                await _store_provider.aclose()
            except Exception:  # noqa: BLE001
                pass
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
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
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
    "tenant_resolver", "load_system_prompt", "load_history", "summarize",
    "vision", "detect_intent", "lead_memory",
    "retrieve_for_close", "retrieve_for_respond",
    "close_sale", "respond", "greeting", "objection", "follow_up",
    "flow_executor", "tools", "supervisor", "strategist",
    "persist", "send",
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

def _redact_phone(phone: str) -> str:
    """Privacy: log apenas últimos 4 dígitos (LGPD/GDPR compliance)."""
    if not phone or len(phone) < 4:
        return "***"
    return f"***{phone[-4:]}"


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

    # Log TODO webhook recebido — diagnóstico volume
    log.info("[webhook] event=%s instance=%s", event, instance)

    if event == "connection.update":
        state = (body.get("data") or {}).get("state") or body.get("state")
        log.info("[conn] %s state=%s", instance, state)
        return {"ok": True, "connectionState": state}

    if event == "presence.update":
        return {"ok": True, "skipped": "presence"}

    if event != "messages.upsert":
        log.info("[webhook] skip event=%s", event)
        return {"ok": True, "skipped": event}

    data = body.get("data") or {}
    key = data.get("key") or {}
    message = data.get("message") or {}

    if key.get("fromMe"):
        log.info("[webhook] skip fromMe %s", instance)
        return {"ok": True, "skipped": "fromMe"}

    remote_jid: str = key.get("remoteJid") or ""
    if remote_jid.endswith("@g.us") or "@broadcast" in remote_jid:
        log.info("[webhook] skip group/broadcast jid=%s", _redact_phone(remote_jid))
        return {"ok": True, "skipped": "group/broadcast"}

    phone = remote_jid.replace("@s.whatsapp.net", "").replace("@g.us", "")
    if not re.fullmatch(r"\d{10,15}", phone or ""):
        log.info("[webhook] skip invalid phone=%s jid=%s", phone, remote_jid)
        return {"ok": True, "skipped": "invalid phone"}

    text = _extract_text(message)
    media = _extract_media(message, data)
    if not text and not media:
        log.info("[webhook] skip no content phone=%s", phone)
        return {"ok": True, "skipped": "no content"}

    message_id = key.get("id") or ""
    if message_id:
        first = await redis.mark_message_processed(instance, phone, message_id)
        if not first:
            log.info("[webhook] skip duplicate phone=%s msg_id=%s", phone, message_id)
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

    # Empilha messageId no buffer de unread — bot vai marcar TODAS antes de responder
    # (cliente manda rajada rápida → todas viram ✓✓ azul juntas, não só a última)
    if message_id:
        try:
            await redis.push_unread(instance, phone, message_id)
        except Exception:  # noqa: BLE001
            pass

    # Coalescing de rajada (anti-spam reply): push pro inbox-per-lead em vez
    # de enfileirar direto. Drainer loop combina msgs que chegam em rajada
    # dentro de INBOX_DEBOUNCE_S e enfileira PAYLOAD ÚNICO com user_message
    # concatenado. Bot responde TUDO em 1 turno (max 4 bolhas).
    inbox_item = {
        "ts": time.time(),
        "text": user_message,
        "message_id": message_id,
        "media_mime": media["mime"] if media else None,
        "media_base64": media["base64"] if media else None,
        "push_name": push_name,
        "project_id_hint": project_id_hint,
    }
    inbox_size = await redis.push_inbox(instance, phone, inbox_item)
    log.info("[inbox] %s/%s pushed (size=%d)", instance, phone, inbox_size)
    return {"ok": True, "queued": False, "inboxed": True, "inbox_size": inbox_size}


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
# Inbox drainer — coalesce rajadas (anti-spam reply)
# Lead manda 3 msgs em sequência → bot espera, junta, responde TUDO em 1 turno.
# ────────────────────────────────────────────────────────────────────

# Tempo em segundos que o inbox precisa ficar IDLE (sem nova msg) pra ser drenado.
# 6s ~ pausa típica de digitação. Menor = arrisca cortar burst. Maior = lentidão.
INBOX_DEBOUNCE_S = float(os.getenv("INBOX_DEBOUNCE_S", "6"))
# Safety: tempo MÁX que msg pode ficar no inbox (mesmo se user continua mandando).
# Após esse limite, força drain — evita lead spammer travar resposta indefinidamente.
INBOX_MAX_AGE_S = float(os.getenv("INBOX_MAX_AGE_S", "20"))
# Drain imediato se inbox atingir esse tamanho (rajada muito grande).
INBOX_MAX_SIZE = int(os.getenv("INBOX_MAX_SIZE", "8"))
# Intervalo de varredura do drainer.
INBOX_SCAN_INTERVAL_S = float(os.getenv("INBOX_SCAN_INTERVAL_S", "1.5"))


async def _inbox_drainer_loop() -> None:
    """
    Smart scheduler — varre inboxes prontos, ranqueia por (HOT lane, wait_time),
    drena na ordem prioritária e enfileira combined payload com flag `hot`.

    Pra cada inbox:
      - idle > INBOX_DEBOUNCE_S → pronto pra drain (rajada terminou)
      - age > INBOX_MAX_AGE_S → força drain (safety)
      - size >= INBOX_MAX_SIZE → drain imediato

    Ranking dos PRONTOS:
      1. HOT lane (lead em estagio "preco"/"fechamento") tem prioridade sempre
      2. Empata? maior wait_time (último resposta do bot foi há mais tempo)

    HOT-lane safety: lead na HOT lane >10min sem fechar volta pra NORMAL (esfriou).
    """
    while not _worker_stop_evt.is_set():
        try:
            try:
                inboxes = await redis.list_active_inboxes()
            except Exception as exc:  # noqa: BLE001
                log.warning("[inbox-drainer] list_active falhou: %s", exc)
                inboxes = []

            now = time.time()
            # 1ª passada: filtra apenas inboxes prontos pra drain + coleta priority info
            ready: list[dict[str, Any]] = []
            for instance, phone in inboxes:
                try:
                    newest_ts = await redis.peek_inbox_newest_ts(instance, phone)
                    if newest_ts is None:
                        continue
                    oldest_ts = await redis.peek_inbox_oldest_ts(instance, phone)
                    size = await redis.inbox_length(instance, phone)

                    idle = now - newest_ts
                    age = (now - oldest_ts) if oldest_ts else 0
                    should_drain = (
                        idle >= INBOX_DEBOUNCE_S
                        or age >= INBOX_MAX_AGE_S
                        or size >= INBOX_MAX_SIZE
                    )
                    if not should_drain:
                        continue

                    # Priority info
                    estagio = await redis.get_lead_stage(instance, phone) or ""
                    last_bot_ts = await redis.get_last_bot_reply_ts(instance, phone) or 0.0
                    wait_time = (now - last_bot_ts) if last_bot_ts else 99999.0

                    # HOT lane com safety: se está há >10min na HOT sem fechar, esfria
                    hot = estagio in HOT_STAGES and wait_time < 600.0

                    ready.append({
                        "instance": instance,
                        "phone": phone,
                        "estagio": estagio,
                        "wait_time": wait_time,
                        "hot": hot,
                        "size": size,
                        "idle": idle,
                        "age": age,
                    })
                except Exception as exc:  # noqa: BLE001
                    log.warning("[inbox-drainer] inspect %s/%s: %s", instance, phone, exc)
                    continue

            # 2ª passada: ranqueia (HOT primeiro, depois maior wait_time)
            ready.sort(key=lambda r: (not r["hot"], -r["wait_time"]))

            # 3ª passada: drena na ordem prioritária + enfileira
            for r in ready:
                instance, phone = r["instance"], r["phone"]
                try:
                    items = await redis.drain_inbox(instance, phone)
                    if not items:
                        continue

                    # Combina texto FIFO + agrega media + extrai meta
                    texts: list[str] = []
                    last_media_mime: str | None = None
                    last_media_b64: str | None = None
                    last_message_id: str = ""
                    push_name = ""
                    project_id_hint = ""
                    for it in items:
                        t = (it.get("text") or "").strip()
                        if t:
                            texts.append(t)
                        if it.get("media_mime"):
                            last_media_mime = it["media_mime"]
                            last_media_b64 = it.get("media_base64")
                        if it.get("message_id"):
                            last_message_id = it["message_id"]
                        if it.get("push_name"):
                            push_name = it["push_name"]
                        if it.get("project_id_hint"):
                            project_id_hint = it["project_id_hint"]

                    combined_text = "\n".join(texts) if texts else "[mídia]"

                    payload = {
                        "kind": "inbound",
                        "instance": instance,
                        "phone": phone,
                        "message_id": last_message_id,
                        "enqueued_at": time.time(),
                        "batch_size": len(items),
                        "hot": r["hot"],
                        "estagio": r["estagio"],
                        "wait_time": r["wait_time"],
                        "initial_state": {
                            "project_id": project_id_hint or os.getenv("DEFAULT_PROJECT_ID", "padrao"),
                            "instance_name": instance,
                            "phone": phone,
                            "push_name": push_name,
                            "user_message": combined_text,
                            "media_mime": last_media_mime,
                            "media_base64": last_media_b64,
                            "message_id": last_message_id,
                            "messages": [],
                        },
                    }
                    qsize = await redis.enqueue_message(payload)
                    log.info(
                        "[scheduler] %s/%s drenou %d msgs hot=%s wait=%.0fs estagio=%s (idle=%.1fs size=%d) → queue=%d",
                        instance, phone, len(items), r["hot"], r["wait_time"], r["estagio"] or "?",
                        r["idle"], r["size"], qsize,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("[inbox-drainer] erro %s/%s: %s", instance, phone, exc)
                    continue
        except asyncio.CancelledError:
            log.info("[inbox-drainer] cancelado")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("[inbox-drainer] loop erro: %s", exc)

        try:
            await asyncio.wait_for(_worker_stop_evt.wait(), timeout=INBOX_SCAN_INTERVAL_S)
            return  # stop_evt setado
        except asyncio.TimeoutError:
            continue


# ────────────────────────────────────────────────────────────────────
# FIFO worker — drena queue 1 mensagem por vez (anti-ban WhatsApp)
# ────────────────────────────────────────────────────────────────────

async def _queue_worker_loop() -> None:
    """
    Smart worker — drena queue, aplica delay gaussian adaptativo entre msgs.

    Delay sempre aplicado entre msgs (não só leads diferentes — bot revezando):
      - HOT lane: 45-75s (lead fechando, urgência)
      - Calmo: 75-150s
      - Normal: 60-120s
      - Pico: 45-90s
    """
    backoff = QUEUE_POLL_INTERVAL_S
    msgs_processed = 0  # contador pra detectar 1ª msg (sem delay inicial)
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
            is_hot = bool(payload.get("hot"))

            # Smart delay — entre TODA msg (não só leads diferentes). Bot revezando.
            # Skip apenas 1ª msg da sessão (sem warm-up).
            if INTER_LEAD_DELAY_ENABLED and msgs_processed > 0:
                qsize_now = await redis.queue_length()
                delay = _calc_inter_lead_delay(qsize_now, hot=is_hot)
                log.info(
                    "[scheduler] delay %.1fs (qsize=%d hot=%s phone=%s)",
                    delay, qsize_now, is_hot, current_phone,
                )
                try:
                    await asyncio.wait_for(_worker_stop_evt.wait(), timeout=delay)
                    return  # stop_evt durante sleep
                except asyncio.TimeoutError:
                    pass

            try:
                await _process_queued(payload)
            finally:
                msgs_processed += 1
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
    got_lock = await redis.acquire_lock(instance, phone, ttl_seconds=60)
    if not got_lock:
        # Smart-lock defer: lock ainda detido → coloca msg no FINAL da queue
        # (lado de entrada = sai por último). Processa outros leads enquanto
        # esse lead "descansa". Cap em 5 defers — depois descarta (lead provavelmente
        # tem worker travado ou lock zumbi).
        defer_count = int(payload.get("defer_count") or 0)
        if defer_count >= 5:
            log.warning(
                "[queue] %s/%s defer_count=%d exceeded — descartando msg (lock zumbi?)",
                instance, phone, defer_count,
            )
            # Tenta forçar release do lock (zumbi) pra próxima msg dele desbloquear
            try:
                await redis.release_lock(instance, phone)
            except Exception:  # noqa: BLE001
                pass
            return
        log.info(
            "[queue] lock_held %s/%s — defer pro fim (defer_count=%d, processando outros)",
            instance, phone, defer_count + 1,
        )
        await redis.defer_message(payload)
        # Pequeno sleep pra evitar burn loop se queue tem só esta msg
        await asyncio.sleep(0.5)
        return

    try:
        # Propaga batch_size do payload pro initial_state — respond_node usa
        # pra ajustar prompt ("user mandou N msgs, responde tudo em 4 bolhas")
        batch_size = int(payload.get("batch_size") or 1)
        if batch_size > 1:
            initial_state["batch_size"] = batch_size

        thread_id = thread_id_for(project_id_hint, instance, phone)
        final_state = await _run_graph_streaming(initial_state, thread_id)
        log.info(
            "[queue] processed %s %s/%s intent=%s sent=%s",
            kind, instance, phone,
            final_state.get("intent"),
            final_state.get("sent_count", 0),
        )

        # Smart scheduler tracking: registra estágio quick-access + last_bot_reply
        try:
            lead_facts = final_state.get("lead_facts") or {}
            estagio = lead_facts.get("estagio")
            if estagio:
                await redis.set_lead_stage(instance, phone, estagio)
            if final_state.get("sent_count", 0) > 0 or final_state.get("sent"):
                await redis.set_last_bot_reply_ts(instance, phone)
                # Chip quota tracking (anti-ban hard cap)
                await redis.mark_chip_first_use(instance)
                qcount = await redis.increment_chip_quota(instance)
                age_days = await redis.get_chip_age_days(instance)
                soft_cap = (
                    50 if age_days < 7 else
                    200 if age_days < 30 else
                    500
                )
                if qcount >= soft_cap:
                    log.warning(
                        "[chip-quota] %s atingiu cap diário (%d/%d, age=%dd) — considere outro chip",
                        instance, qcount, soft_cap, age_days,
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning("[scheduler] tracking falhou %s/%s: %s", instance, phone, exc)

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
        "store": _store_provider.kind if _store_provider else "none",
        "qstash": qstash.enabled,
        "queue_size": qsize,
        "worker_alive": worker_alive,
    }


@app.post("/webhook/reset-admin")
async def webhook_reset_admin(request: Request) -> dict:
    """
    Reset/criar admin via WEBHOOK_SECRET (sem precisar login painel).

    Use quando esqueceu senha do painel. Auth = mesma do webhook normal.

      curl -X POST -H "apikey: $WEBHOOK_SECRET" \\
        -H "Content-Type: application/json" \\
        -d '{"email":"admin@local","password":"nova-senha-forte"}' \\
        https://app.up.railway.app/webhook/reset-admin
    """
    _check_auth(request)
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()
    display_name = (body.get("display_name") or email.split("@")[0]).strip()

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="email obrigatório (formato user@host)")
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="password obrigatório (>= 6 chars)")

    sb_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    # SECURITY: SERVICE_KEY only — ANON_KEY não deve poder escrever em admin_users.
    # Se ANON_KEY tivesse esse poder, leak da ANON_KEY (que pode ser exposta no
    # frontend em alguns setups) daria controle total ao bot.
    sb_key = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
    if not sb_url or not sb_key:
        raise HTTPException(status_code=503, detail="SUPABASE_URL/SUPABASE_SERVICE_KEY ausentes")

    from panel.auth import hash_password
    import httpx
    pwd_hash = hash_password(password)

    async with httpx.AsyncClient(timeout=10.0) as c:
        # Tenta UPDATE primeiro
        r = await c.patch(
            f"{sb_url}/rest/v1/admin_users",
            params={"email": f"eq.{email}"},
            headers={
                "apikey": sb_key,
                "Authorization": f"Bearer {sb_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json={"password_hash": pwd_hash, "display_name": display_name},
        )
        if not r.is_success:
            raise HTTPException(status_code=502, detail=f"Supabase erro: {r.text[:200]}")
        existing = r.json() or []
        if existing:
            log.info("[reset-admin] update senha %s", email)
            return {"ok": True, "action": "updated", "email": email, "id": existing[0].get("id")}

        # INSERT novo
        r2 = await c.post(
            f"{sb_url}/rest/v1/admin_users",
            headers={
                "apikey": sb_key,
                "Authorization": f"Bearer {sb_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json={
                "email": email,
                "password_hash": pwd_hash,
                "display_name": display_name,
                "project_ids": [],
            },
        )
        if not r2.is_success:
            raise HTTPException(status_code=502, detail=f"Supabase erro: {r2.text[:200]}")
        created = r2.json() or []
        log.info("[reset-admin] criou %s", email)
        return {"ok": True, "action": "created", "email": email, "id": (created[0].get("id") if created else None)}


@app.get("/webhook/self-check")
async def webhook_self_check(request: Request) -> dict:
    """
    Diagnóstico self-service — auth via WEBHOOK_SECRET (sem login admin).

    Lista todas instances Evolution, mostra: status conexão + webhook URL atual
    + se URL bate com a expected. Auto-fixa instances com webhook errado/ausente
    se ?fix=1 query param.

    Use:
      curl -H "apikey: $WEBHOOK_SECRET" \\
        https://app.up.railway.app/webhook/self-check
      curl -H "apikey: $WEBHOOK_SECRET" \\
        https://app.up.railway.app/webhook/self-check?fix=1
    """
    _check_auth(request)
    base_url = (os.getenv("EVOLUTION_API_URL") or "").rstrip("/")
    api_key = os.getenv("EVOLUTION_API_KEY") or ""
    if not base_url or not api_key:
        return {"ok": False, "error": "EVOLUTION_API_URL / EVOLUTION_API_KEY missing"}

    pub_raw = (os.getenv("PUBLIC_BASE_URL") or os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip()
    if pub_raw and not pub_raw.startswith("http"):
        pub_raw = "https://" + pub_raw
    pub_url = pub_raw.rstrip("/")
    expected_webhook = f"{pub_url}/webhook/evolution" if pub_url else None
    fix = request.query_params.get("fix") == "1"

    import httpx
    report: dict[str, Any] = {
        "ok": True,
        "expected_webhook": expected_webhook,
        "fix_mode": fix,
        "instances": [],
    }
    secret = (os.getenv("WEBHOOK_SECRET") or "").strip()

    async with httpx.AsyncClient(timeout=15.0) as c:
        try:
            r = await c.get(f"{base_url}/instance/fetchInstances", headers={"apikey": api_key})
            r.raise_for_status()
            payload = r.json() or []
        except httpx.HTTPError as e:
            return {"ok": False, "error": f"Evolution unreachable: {e}"[:200]}

        for item in payload:
            inst = item.get("instance") if isinstance(item, dict) and "instance" in item else item
            if not isinstance(inst, dict):
                continue
            name = inst.get("instanceName") or inst.get("name") or ""
            status = inst.get("status") or inst.get("state") or "unknown"
            info: dict[str, Any] = {"name": name, "status": status}

            # Webhook config atual
            try:
                wr = await c.get(f"{base_url}/webhook/find/{name}", headers={"apikey": api_key})
                if wr.is_success and wr.content:
                    wd = wr.json() or {}
                    info["webhook_url"] = wd.get("url") or ""
                    info["webhook_enabled"] = wd.get("enabled", False)
                else:
                    info["webhook_url"] = ""
                    info["webhook_enabled"] = False
            except httpx.HTTPError as e:
                info["webhook_url"] = f"<error: {str(e)[:60]}>"

            info["webhook_match"] = bool(
                expected_webhook
                and info.get("webhook_url")
                and info["webhook_url"].rstrip("/") == expected_webhook.rstrip("/")
                and info.get("webhook_enabled")
            )

            # Auto-fix se ?fix=1 e webhook errado
            if fix and not info["webhook_match"] and expected_webhook and secret:
                try:
                    fr = await c.post(
                        f"{base_url}/webhook/set/{name}",
                        headers={"apikey": api_key, "Content-Type": "application/json"},
                        json={
                            "webhook": {
                                "url": expected_webhook,
                                "enabled": True,
                                "webhookByEvents": False,
                                "webhookBase64": True,
                                "headers": {"apikey": secret, "Content-Type": "application/json"},
                                "events": ["MESSAGES_UPSERT", "MESSAGES_UPDATE", "CONNECTION_UPDATE", "SEND_MESSAGE"],
                            }
                        },
                    )
                    info["fix_applied"] = fr.is_success
                    if not fr.is_success:
                        info["fix_error"] = fr.text[:200]
                except httpx.HTTPError as e:
                    info["fix_applied"] = False
                    info["fix_error"] = str(e)[:200]

            report["instances"].append(info)

    # Resumo no topo
    report["summary"] = {
        "total": len(report["instances"]),
        "connected": sum(1 for i in report["instances"] if str(i.get("status", "")).lower() in ("open", "connected")),
        "webhook_ok": sum(1 for i in report["instances"] if i.get("webhook_match")),
        "needs_fix": sum(1 for i in report["instances"] if not i.get("webhook_match")),
    }
    return report


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD")),
    )
