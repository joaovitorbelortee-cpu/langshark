"""
Sub-router do painel admin — endpoints JSON sob /api/admin/*.

F1: GETs read-only.
F2: + auth (login/logout/me).
F3-F5: PATCH/POST/DELETE.
"""
from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, File, HTTPException, Response, UploadFile

from agent.tools import EvolutionClient
from panel.auth import (
    authenticate,
    clear_session_cookie,
    create_token,
    require_admin,
    set_session_cookie,
)
from panel.repos import AIModelsCatalogRepo, AuditLogRepo, ProjectConfigRepo


admin_router = APIRouter()
_project_repo = ProjectConfigRepo()
_models_repo = AIModelsCatalogRepo()
_audit_repo = AuditLogRepo()
_evo = EvolutionClient()


async def _audit(user: dict[str, Any], action: str, target_type: str, target_id: str, **metadata: Any) -> None:
    """Helper: registra mutação no audit_log. Best-effort, falha silenciosa."""
    await _audit_repo.write(
        actor_id=str(user.get("id", "")),
        actor_email=user.get("email", ""),
        action=action,
        target_type=target_type,
        target_id=target_id,
        metadata=metadata or None,
    )


# ────────────────────────────────────────────────────────────────────
# Instances list cache — 10s TTL pra reduzir chamadas Evolution
# ────────────────────────────────────────────────────────────────────

import time as _time  # noqa: E402

_instances_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_INSTANCES_TTL = 10.0


@admin_router.get("/health")
async def health() -> dict[str, Any]:
    """Health do sub-app admin (separado do health geral do bot)."""
    return {"ok": True, "scope": "admin"}


# ────────────────────────────────────────────────────────────────────
# Auth (F2)
# ────────────────────────────────────────────────────────────────────

@admin_router.post("/auth/login")
async def auth_login(
    response: Response,
    body: dict = Body(...),
) -> dict[str, Any]:
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="email e password obrigatorios")
    user = await authenticate(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciais invalidas")
    token = create_token(user)
    set_session_cookie(response, token)
    return {
        "ok": True,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "display_name": user.get("display_name"),
            "project_ids": user.get("project_ids", []),
        },
    }


@admin_router.post("/auth/logout")
async def auth_logout(response: Response) -> dict[str, Any]:
    clear_session_cookie(response)
    return {"ok": True}


@admin_router.get("/me")
async def me(
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user.get("display_name"),
        "project_ids": user.get("project_ids", []),
        "last_login_at": user.get("last_login_at"),
    }


# ────────────────────────────────────────────────────────────────────
# Projects (read-only F1)
# ────────────────────────────────────────────────────────────────────

def _user_can_access_project(user: dict[str, Any], project_id: str) -> bool:
    """Multi-tenant guard: user.project_ids vazio = acesso total (admin global)."""
    allowed = user.get("project_ids") or []
    return (not allowed) or (project_id in allowed)


@admin_router.get("/projects")
async def list_projects(
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> list[dict[str, Any]]:
    rows = await _project_repo.list()
    allowed = user.get("project_ids") or []
    if allowed:
        rows = [r for r in rows if r["project_id"] in allowed]
    # Resposta enxuta — detail tem o resto
    return [
        {
            "project_id":  r["project_id"],
            "display_name": r.get("display_name") or r["project_id"],
            "agent_name":   r.get("agent_name", ""),
            "ai_model":     r.get("ai_model", ""),
            "is_active":    r.get("is_active", True),
            "updated_at":   r.get("updated_at"),
        }
        for r in rows
    ]


@admin_router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    cfg = await _project_repo.fetch(project_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return cfg


# ────────────────────────────────────────────────────────────────────
# Project config — edição (F3)
# ────────────────────────────────────────────────────────────────────

def _invalidate_cache(project_id: str) -> None:
    """Invalida cache local + emite pub/sub Redis (cross-worker em F7)."""
    try:
        from panel.cache import get_project_config_cache
        get_project_config_cache().invalidate(project_id)
    except Exception:
        pass


@admin_router.patch("/projects/{project_id}/sections/{section_key}")
async def patch_section(
    project_id: str,
    section_key: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    valid_keys = {"company_info", "prices", "parameters", "priority_situations", "knowledge_base"}
    if section_key not in valid_keys:
        raise HTTPException(status_code=400, detail=f"section_key invalido. validos: {valid_keys}")
    content = body.get("content") or ""
    if len(content) > 7000:
        raise HTTPException(status_code=400, detail="Conteudo > 7000 chars")
    cfg = await _project_repo.patch_section(project_id, section_key, content)
    _invalidate_cache(project_id)
    await _audit(user, "section.patch", "project_config", project_id,
                 section_key=section_key, size=len(content))
    return {"ok": True, "section": section_key, "size": len(content), "config": cfg}


@admin_router.patch("/projects/{project_id}/config")
async def patch_config(
    project_id: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    allowed = {"agent_name", "ai_model", "ai_temperature", "ai_max_tokens", "is_active", "display_name", "followup_enabled"}
    payload = {k: v for k, v in body.items() if k in allowed}
    if not payload:
        raise HTTPException(status_code=400, detail="Nenhum campo valido pra atualizar")
    cfg = await _project_repo.patch(project_id, payload)
    _invalidate_cache(project_id)
    await _audit(user, "config.patch", "project_config", project_id, fields=list(payload.keys()))
    return {"ok": True, "updated": list(payload.keys()), "config": cfg}


# ────────────────────────────────────────────────────────────────────
# AI Models catalog
# ────────────────────────────────────────────────────────────────────

@admin_router.get("/models")
async def list_models(
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> list[dict[str, Any]]:
    return await _models_repo.list(only_active=True)


# ────────────────────────────────────────────────────────────────────
# Instances (proxy Evolution + binding instance_projects)
# ────────────────────────────────────────────────────────────────────

@admin_router.get("/instances")
async def list_instances(
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> list[dict[str, Any]]:
    """
    Lista instâncias da Evolution API + status de conexão.
    Cache 10s reduz chamadas se UI faz polling.
    """
    base_url = os.getenv("EVOLUTION_API_URL", "").rstrip("/")
    api_key = os.getenv("EVOLUTION_API_KEY", "")
    if not base_url or not api_key:
        return []

    now = _time.monotonic()
    cached = _instances_cache.get(base_url)
    if cached and cached[1] > now:
        return cached[0]

    import httpx
    last_exc: Exception | None = None
    payload: list[Any] = []
    # Retry leve: até 2 tentativas com backoff curto
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                r = await c.get(
                    f"{base_url}/instance/fetchInstances",
                    headers={"apikey": api_key},
                )
                r.raise_for_status()
                payload = r.json() or []
                last_exc = None
                break
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < 2:
                import asyncio
                await asyncio.sleep(0.5)
    if last_exc is not None:
        # Serve cache stale se houver, senão 502.
        if cached:
            return cached[0]
        raise HTTPException(status_code=502, detail=f"Evolution unreachable: {last_exc}")

    out: list[dict[str, Any]] = []
    for item in payload:
        # Evolution API v2 retorna estrutura aninhada {instance: {...}}
        inst = item.get("instance") if isinstance(item, dict) and "instance" in item else item
        if not isinstance(inst, dict):
            continue
        out.append({
            "instance_name": inst.get("instanceName") or inst.get("name"),
            "status":        inst.get("status") or inst.get("state") or "unknown",
            "owner":         inst.get("owner"),
            "profile_name":  inst.get("profileName"),
            "profile_pic":   inst.get("profilePicUrl"),
            "integration":   inst.get("integration", "WHATSAPP-BAILEYS"),
        })
    _instances_cache[base_url] = (out, now + _INSTANCES_TTL)
    return out


def _evolution_creds() -> tuple[str, str]:
    return os.getenv("EVOLUTION_API_URL", "").rstrip("/"), os.getenv("EVOLUTION_API_KEY", "")


@admin_router.post("/instances")
async def create_instance(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    """Cria instância Evolution + opcionalmente bind a project_id."""
    name = (body.get("instance_name") or "").strip()
    project_id = (body.get("project_id") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="instance_name obrigatorio")
    if project_id and not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden: project_id fora do escopo")
    base_url, api_key = _evolution_creds()
    if not base_url:
        raise HTTPException(status_code=503, detail="EVOLUTION_API_URL ausente")
    import httpx
    async with httpx.AsyncClient(timeout=20.0) as c:
        # Cria instância com settings humanos já habilitados:
        # - readMessages: bot envia ✓✓ azul (read receipts)
        # - alwaysOnline: instance fica online → delivery ✓✓ rápido
        # - rejectCall: rejeita chamadas (bot não atende voz)
        # - groupsIgnore: ignora msgs de grupo (anti-spam)
        r = await c.post(
            f"{base_url}/instance/create",
            headers={"apikey": api_key, "Content-Type": "application/json"},
            json={
                "instanceName": name,
                "qrcode": True,
                "integration": "WHATSAPP-BAILEYS",
                "readMessages": True,
                "readStatus": False,
                "alwaysOnline": True,
                "rejectCall": True,
                "msgCall": "No momento não posso atender chamadas. Pode me chamar no chat 🙏",
                "groupsIgnore": True,
                "syncFullHistory": False,
            },
        )
        if not r.is_success:
            raise HTTPException(status_code=r.status_code, detail=r.text[:200])
        evo_payload = r.json()
        # Garante settings (alguns deploys ignoram campos no create — força via /settings/set)
        try:
            await c.post(
                f"{base_url}/settings/set/{name}",
                headers={"apikey": api_key, "Content-Type": "application/json"},
                json={
                    "readMessages": True,
                    "alwaysOnline": True,
                    "rejectCall": True,
                    "groupsIgnore": True,
                    "msgCall": "No momento não posso atender chamadas. Pode me chamar no chat 🙏",
                },
            )
        except httpx.HTTPError:
            pass  # best-effort

        # CRÍTICO: configura webhook pra Evolution mandar eventos pro nosso /webhook/evolution
        # Sem isso, msgs do WhatsApp NÃO chegam no bot.
        pub_url = _public_base_url()
        secret = (os.getenv("WEBHOOK_SECRET") or "").strip()
        if pub_url and secret:
            try:
                await c.post(
                    f"{base_url}/webhook/set/{name}",
                    headers={"apikey": api_key, "Content-Type": "application/json"},
                    json={
                        "webhook": {
                            "url": f"{pub_url}/webhook/evolution",
                            "enabled": True,
                            "webhookByEvents": False,
                            "webhookBase64": True,
                            "headers": {"apikey": secret, "Content-Type": "application/json"},
                            "events": ["MESSAGES_UPSERT", "MESSAGES_UPDATE", "CONNECTION_UPDATE", "SEND_MESSAGE"],
                        }
                    },
                )
            except httpx.HTTPError:
                pass  # best-effort
    # Bind ao projeto (tabela instance_projects)
    if project_id:
        from panel.repos import _supabase_creds, _headers
        url, _ = _supabase_creds()
        async with httpx.AsyncClient(timeout=5.0) as c:
            await c.post(
                f"{url}/rest/v1/instance_projects",
                headers={**_headers(), "Prefer": "resolution=merge-duplicates"},
                json={"instance_name": name, "project_id": project_id},
            )
    # Invalida cache pra próxima list ver a nova instância
    _instances_cache.clear()
    await _audit(user, "instance.create", "evolution_instance", name, project_id=project_id)
    qr = (evo_payload.get("qrcode") or {}).get("base64")
    return {"ok": True, "instance_name": name, "qr_base64": qr, "evolution": evo_payload}


@admin_router.delete("/instances/{instance_name}")
async def delete_instance(
    instance_name: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    base_url, api_key = _evolution_creds()
    if not base_url:
        raise HTTPException(status_code=503, detail="EVOLUTION_API_URL ausente")
    import httpx
    async with httpx.AsyncClient(timeout=20.0) as c:
        # Logout primeiro (ignora erro), depois delete
        await c.delete(f"{base_url}/instance/logout/{instance_name}", headers={"apikey": api_key})
        r = await c.delete(f"{base_url}/instance/delete/{instance_name}", headers={"apikey": api_key})
        if r.status_code not in (200, 404):
            raise HTTPException(status_code=r.status_code, detail=r.text[:200])
    _instances_cache.clear()
    await _audit(user, "instance.delete", "evolution_instance", instance_name)
    return {"ok": True, "deleted": instance_name}


@admin_router.get("/instances/{instance_name}/qr")
async def get_instance_qr(
    instance_name: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    base_url, api_key = _evolution_creds()
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(f"{base_url}/instance/connect/{instance_name}", headers={"apikey": api_key})
        if not r.is_success:
            raise HTTPException(status_code=r.status_code, detail=r.text[:200])
        d = r.json()
    return {"ok": True, "qr_base64": d.get("base64"), "code": d.get("code")}


@admin_router.post("/instances/{instance_name}/enable-read-receipts")
async def enable_read_receipts(
    instance_name: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    """
    Liga settings humanos numa instância EXISTENTE:
      - readMessages: True   → ✓✓ azul aparece
      - alwaysOnline: True   → delivery ✓✓ rápido
      - rejectCall: True     → rejeita chamadas
      - groupsIgnore: True   → ignora grupos

    Necessário em instâncias antigas criadas antes do default-on.
    """
    base_url, api_key = _evolution_creds()
    if not base_url:
        raise HTTPException(status_code=503, detail="EVOLUTION_API_URL ausente")
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(
            f"{base_url}/settings/set/{instance_name}",
            headers={"apikey": api_key, "Content-Type": "application/json"},
            json={
                "readMessages": True,
                "readStatus": False,
                "alwaysOnline": True,
                "rejectCall": True,
                "msgCall": "No momento não posso atender chamadas. Pode me chamar no chat 🙏",
                "groupsIgnore": True,
                "syncFullHistory": False,
            },
        )
        if not r.is_success:
            raise HTTPException(status_code=r.status_code, detail=r.text[:300])
        payload = r.json() if r.content else {}
    await _audit(user, "instance.enable_read_receipts", "evolution_instance", instance_name)
    return {"ok": True, "instance": instance_name, "evolution": payload}


@admin_router.get("/instances/{instance_name}/settings")
async def get_instance_settings(
    instance_name: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    """GET settings atuais da instância — útil pra debug."""
    base_url, api_key = _evolution_creds()
    if not base_url:
        raise HTTPException(status_code=503, detail="EVOLUTION_API_URL ausente")
    import httpx
    async with httpx.AsyncClient(timeout=8.0) as c:
        r = await c.get(
            f"{base_url}/settings/find/{instance_name}",
            headers={"apikey": api_key},
        )
        if not r.is_success:
            return {"ok": False, "status": r.status_code}
        return r.json() if r.content else {}


@admin_router.get("/instances/{instance_name}/status")
async def get_instance_status(
    instance_name: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    base_url, api_key = _evolution_creds()
    import httpx
    async with httpx.AsyncClient(timeout=8.0) as c:
        r = await c.get(f"{base_url}/instance/connectionState/{instance_name}", headers={"apikey": api_key})
        if not r.is_success:
            return {"state": "unknown"}
        d = r.json()
    inst = d.get("instance") or {}
    return {"state": inst.get("state") or "unknown"}


def _public_base_url() -> str:
    """Resolve URL pública do app pra config de webhook na Evolution."""
    raw = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if not raw:
        raw = (os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip()
    if not raw:
        return ""
    if not raw.startswith("http"):
        raw = "https://" + raw
    return raw.rstrip("/")


@admin_router.post("/instances/{instance_name}/configure-webhook")
async def configure_webhook(
    instance_name: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    """
    Configura Evolution pra mandar eventos pro nosso /webhook/evolution.
    Crítico — sem isso, msgs do WhatsApp não chegam no bot.
    Run em instâncias existentes que foram criadas sem webhook setado.
    """
    base_url, api_key = _evolution_creds()
    if not base_url:
        raise HTTPException(status_code=503, detail="EVOLUTION_API_URL ausente")
    pub_url = _public_base_url()
    if not pub_url:
        raise HTTPException(status_code=503, detail="PUBLIC_BASE_URL ausente — não dá pra resolver webhook URL")
    secret = (os.getenv("WEBHOOK_SECRET") or "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="WEBHOOK_SECRET ausente")

    webhook_url = f"{pub_url}/webhook/evolution"
    import httpx
    payload = {
        "webhook": {
            "url": webhook_url,
            "enabled": True,
            "webhookByEvents": False,
            "webhookBase64": True,  # mídia já vem em base64
            "headers": {
                "apikey": secret,  # autentica como WEBHOOK_SECRET
                "Content-Type": "application/json",
            },
            "events": [
                "MESSAGES_UPSERT",   # mensagens recebidas
                "MESSAGES_UPDATE",   # status (read/delivered)
                "CONNECTION_UPDATE", # status conexão
                "SEND_MESSAGE",      # confirmação envio
            ],
        }
    }
    async with httpx.AsyncClient(timeout=10.0) as c:
        # Evolution v2 endpoint: /webhook/set/{instance}
        r = await c.post(
            f"{base_url}/webhook/set/{instance_name}",
            headers={"apikey": api_key, "Content-Type": "application/json"},
            json=payload,
        )
        if not r.is_success:
            raise HTTPException(status_code=r.status_code, detail=r.text[:300])
        result = r.json() if r.content else {}
    await _audit(user, "instance.configure_webhook", "evolution_instance", instance_name, webhook_url=webhook_url)
    return {"ok": True, "instance": instance_name, "webhook_url": webhook_url, "evolution": result}


@admin_router.get("/instances/{instance_name}/diagnose")
async def diagnose_instance(
    instance_name: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    """
    Diagnóstico completo da instance: status conexão + settings + webhook + telefone.
    Use quando bot não responde pra ver o que tá quebrado.
    """
    base_url, api_key = _evolution_creds()
    if not base_url:
        raise HTTPException(status_code=503, detail="EVOLUTION_API_URL ausente")
    import httpx
    report: dict[str, Any] = {"instance_name": instance_name}
    async with httpx.AsyncClient(timeout=8.0) as c:
        # 1. Estado da conexão
        try:
            r = await c.get(f"{base_url}/instance/connectionState/{instance_name}", headers={"apikey": api_key})
            report["connection"] = r.json() if r.is_success else {"error": r.status_code}
        except httpx.HTTPError as e:
            report["connection"] = {"error": str(e)[:120]}
        # 2. Settings
        try:
            r = await c.get(f"{base_url}/settings/find/{instance_name}", headers={"apikey": api_key})
            report["settings"] = r.json() if r.is_success else {"error": r.status_code}
        except httpx.HTTPError as e:
            report["settings"] = {"error": str(e)[:120]}
        # 3. Webhook config
        try:
            r = await c.get(f"{base_url}/webhook/find/{instance_name}", headers={"apikey": api_key})
            report["webhook"] = r.json() if r.is_success else {"error": r.status_code}
        except httpx.HTTPError as e:
            report["webhook"] = {"error": str(e)[:120]}
    # 4. Expected webhook URL
    pub = _public_base_url()
    report["expected_webhook_url"] = f"{pub}/webhook/evolution" if pub else None
    # 5. Webhook URL match?
    actual_url = ((report.get("webhook") or {}).get("url") or "") if isinstance(report.get("webhook"), dict) else ""
    report["webhook_match"] = bool(pub and actual_url and actual_url.startswith(pub))
    return report


@admin_router.post("/debug/test-redis-url")
async def test_redis_url(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    """
    Testa REDIS_URL TCP nativo (rediss://...) antes de setar no Railway env.
    Evita commit-redeploy-debug loop.

    Body: {"url": "rediss://default:senha@host:6379"}
    Retorna: {ok, latency_ms, error, server_info}
    """
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url obrigatória")
    if not url.startswith(("redis://", "rediss://")):
        raise HTTPException(status_code=400, detail="URL deve começar com redis:// ou rediss://")

    try:
        import redis.asyncio as redis_async
        import time as _t
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"redis-py não instalado: {e}")

    started = _t.time()
    try:
        client = redis_async.from_url(url, socket_timeout=8, socket_connect_timeout=8)
        # Ping + info
        await client.ping()
        info_raw = await client.info("server")
        latency_ms = (_t.time() - started) * 1000.0
        await client.aclose()
        srv = {
            "redis_version": info_raw.get("redis_version") if isinstance(info_raw, dict) else None,
            "redis_mode": info_raw.get("redis_mode") if isinstance(info_raw, dict) else None,
        }
        return {
            "ok": True,
            "latency_ms": round(latency_ms, 1),
            "server": srv,
            "msg": "Conexão OK. Pode setar REDIS_URL no Railway com essa URL.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc)[:300],
            "msg": "Falhou. Verifica URL/senha. Não setar no Railway ainda.",
        }


# ────────────────────────────────────────────────────────────────────
# Debug: simular mensagem (testar graph sem WhatsApp)
# ────────────────────────────────────────────────────────────────────

@admin_router.post("/debug/simulate")
async def simulate_message(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    """
    Dispara o graph LangGraph com payload simulado SEM passar pela queue/Evolution.
    Use pra testar prompts/regras enquanto a instance WhatsApp tá offline.

    Body: {"project_id": "padrao", "phone": "5511...", "message": "oi"}

    Retorna:
      - reply (texto que bot mandaria)
      - chunks (mensagens fragmentadas)
      - intent, has_converted, schedule_minutes (tags parsed)
      - final_state (state completo pós-graph)
    """
    project_id = (body.get("project_id") or "padrao").strip()
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    phone = (body.get("phone") or "").strip()
    message = (body.get("message") or "").strip()
    if not phone or not message:
        raise HTTPException(status_code=400, detail="phone e message obrigatórios")

    import main as _main_mod  # import lazy pra evitar ciclo
    if _main_mod._graph_app is None:
        raise HTTPException(status_code=503, detail="Graph não inicializado")

    instance = (body.get("instance_name") or "_debug_").strip()
    initial_state = {
        "project_id": project_id,
        "instance_name": instance,
        "phone": phone,
        "push_name": body.get("push_name") or "Debug User",
        "user_message": message,
        "media_mime": None,
        "media_base64": None,
        "message_id": f"debug-{int(_time.time() * 1000)}",
        "messages": [],
    }
    from agent.checkpointer import thread_id_for
    thread_id = thread_id_for(project_id, instance, phone)
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        # Roda o graph todo (mas send_node é no-op pra _debug_ instance? Não — vai
        # tentar enviar via Evolution. Use instance real OU desliga via flag.)
        # Pra evitar envio: setamos flag no state que send_node respeita.
        initial_state["_debug_no_send"] = True  # send_node ignora
        await _main_mod._graph_app.ainvoke(initial_state, config=cfg)
        snap = await _main_mod._graph_app.aget_state(cfg)
        final_state = dict(snap.values or {}) if snap else {}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Graph erro: {exc}"[:300])

    # Sanitiza state pra serializar (remove BaseMessage objects)
    state_clean: dict[str, Any] = {}
    for k, v in final_state.items():
        if k == "messages":
            state_clean["messages_count"] = len(v) if isinstance(v, list) else 0
            state_clean["last_messages"] = [
                {
                    "type": getattr(m, "type", "?"),
                    "content": str(getattr(m, "content", ""))[:200],
                }
                for m in (v or [])[-4:]
            ]
        else:
            try:
                import json as _json
                _json.dumps(v)
                state_clean[k] = v
            except (TypeError, ValueError):
                state_clean[k] = str(v)[:200]

    await _audit(user, "debug.simulate", "graph", phone, message=message[:80])
    return {
        "ok": True,
        "thread_id": thread_id,
        "reply": final_state.get("reply") or "",
        "chunks": final_state.get("chunks") or [],
        "intent": final_state.get("intent"),
        "has_converted": final_state.get("has_converted"),
        "schedule_minutes": final_state.get("schedule_minutes"),
        "supervisor_review": final_state.get("supervisor_review"),
        "lead_facts": final_state.get("lead_facts"),
        "state": state_clean,
    }


# ────────────────────────────────────────────────────────────────────
# Memory viewers — debug Episodic + Procedural
# ────────────────────────────────────────────────────────────────────

@admin_router.get("/memory/wins")
async def list_wins(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    project_id: str = "padrao",
    limit: int = 20,
) -> dict[str, Any]:
    """Lista episodic wins (conversas que converteram) gravadas no Store."""
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    from agent.store import get_shared_store
    store = get_shared_store()
    if store is None:
        return {"ok": False, "error": "Store não inicializado (REDIS_URL?)", "wins": []}
    try:
        results = await store.asearch((project_id, "wins"), limit=limit)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200], "wins": []}
    wins = []
    for item in results:
        wins.append({
            "key": getattr(item, "key", None),
            "value": getattr(item, "value", None),
            "created_at": str(getattr(item, "created_at", "")),
        })
    return {"ok": True, "count": len(wins), "wins": wins}


@admin_router.get("/memory/lessons")
async def list_lessons(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    project_id: str = "padrao",
    limit: int = 20,
) -> dict[str, Any]:
    """Lista procedural lessons (erros do supervisor) gravadas no Store."""
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    from agent.store import get_shared_store
    store = get_shared_store()
    if store is None:
        return {"ok": False, "error": "Store não inicializado", "lessons": []}
    try:
        results = await store.asearch((project_id, "lessons"), limit=limit)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200], "lessons": []}
    lessons = []
    for item in results:
        lessons.append({
            "key": getattr(item, "key", None),
            "value": getattr(item, "value", None),
            "created_at": str(getattr(item, "created_at", "")),
        })
    return {"ok": True, "count": len(lessons), "lessons": lessons}


@admin_router.delete("/memory/lessons/{key}")
async def delete_lesson(
    key: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
    project_id: str = "padrao",
) -> dict[str, Any]:
    """Remove lesson errada/obsoleta do Store (cleanup manual)."""
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    from agent.store import get_shared_store
    store = get_shared_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Store não inicializado")
    try:
        await store.adelete((project_id, "lessons"), key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:200])
    await _audit(user, "memory.delete_lesson", "store", key)
    return {"ok": True, "deleted": key}


# ────────────────────────────────────────────────────────────────────
# Flows CRUD (F4)
# ────────────────────────────────────────────────────────────────────

from panel.repos import FlowsRepo, ProductsRepo  # noqa: E402

_flows_repo = FlowsRepo()
_products_repo = ProductsRepo()


@admin_router.get("/flows")
async def list_flows(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    project_id: str = "padrao",
) -> list[dict[str, Any]]:
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    return await _flows_repo.list(project_id=project_id)


@admin_router.post("/flows")
async def create_flow(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    project_id = body.get("project_id") or "padrao"
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name obrigatorio")
    row = await _flows_repo.insert({
        "project_id":  project_id,
        "name":        name,
        "description": body.get("description") or "",
        "steps":       body.get("steps") or [],
        "enabled":     body.get("enabled", True),
    })
    # Invalida cache do agent/flows.py — bot precisa enxergar flow novo imediatamente
    try:
        from agent.flows import invalidate_flows_cache
        invalidate_flows_cache(project_id)
    except Exception:
        pass
    await _audit(user, "flow.create", "flow", str(row.get("id", "")), name=name, project_id=project_id)
    return {"ok": True, "flow": row}


@admin_router.patch("/flows/{flow_id}")
async def patch_flow(
    flow_id: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    allowed = {"name", "description", "steps", "enabled"}
    payload = {k: v for k, v in body.items() if k in allowed}
    row = await _flows_repo.patch(flow_id, payload)
    # Cache invalidate — flow editado deve refletir imediatamente
    try:
        from agent.flows import invalidate_flows_cache
        project_id = row.get("project_id") if isinstance(row, dict) else None
        invalidate_flows_cache(project_id)
    except Exception:
        pass
    await _audit(user, "flow.patch", "flow", flow_id, fields=list(payload.keys()))
    return {"ok": True, "flow": row}


@admin_router.delete("/flows/{flow_id}")
async def delete_flow(
    flow_id: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    ok = await _flows_repo.delete(flow_id)
    # Cache invalidate global (não temos project_id do flow_id deletado sem fetch)
    try:
        from agent.flows import invalidate_flows_cache
        invalidate_flows_cache(None)
    except Exception:
        pass
    await _audit(user, "flow.delete", "flow", flow_id)
    return {"ok": ok}


# ────────────────────────────────────────────────────────────────────
# Products CRUD (F4 — Base de Conhecimento)
# ────────────────────────────────────────────────────────────────────

@admin_router.get("/products")
async def list_products(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    project_id: str = "padrao",
) -> list[dict[str, Any]]:
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    return await _products_repo.list(project_id=project_id)


@admin_router.post("/products")
async def create_product(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    project_id = body.get("project_id") or "padrao"
    if not _user_can_access_project(user, project_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name obrigatorio")
    row = await _products_repo.insert({
        "id":          body.get("id"),  # auto-uuid se vazio
        "project_id":  project_id,
        "name":        name,
        "description": body.get("description") or "",
        "price":       body.get("price"),
        "metadata":    body.get("metadata") or {},
    })
    await _audit(user, "product.create", "product", str(row.get("id", "")), name=name, project_id=project_id)
    return {"ok": True, "product": row}


@admin_router.patch("/products/{product_id}")
async def patch_product(
    product_id: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
    body: dict = Body(...),
) -> dict[str, Any]:
    allowed = {"name", "description", "price", "metadata"}
    payload = {k: v for k, v in body.items() if k in allowed}
    row = await _products_repo.patch(product_id, payload)
    await _audit(user, "product.patch", "product", product_id, fields=list(payload.keys()))
    return {"ok": True, "product": row}


@admin_router.delete("/products/{product_id}")
async def delete_product(
    product_id: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    ok = await _products_repo.delete(product_id)
    await _audit(user, "product.delete", "product", product_id)
    return {"ok": ok}


# ────────────────────────────────────────────────────────────────────
# Media upload — bucket Supabase Storage `flow-media` (público).
# Usado pelo flow builder pra anexar imagem/video/audio/documento.
# ────────────────────────────────────────────────────────────────────

MEDIA_MAX_SIZE = 25 * 1024 * 1024  # 25 MB
MEDIA_BUCKET = "flow-media"
MEDIA_ALLOWED_MIME_PREFIXES = ("image/", "video/", "audio/", "application/")


@admin_router.post("/media/upload")
async def upload_media(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Faz upload de mídia pro bucket Supabase `flow-media` (público).
    Retorna URL pública pra usar em step.url de fluxos.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")

    mime = (file.content_type or "application/octet-stream").lower()
    if not any(mime.startswith(p) for p in MEDIA_ALLOWED_MIME_PREFIXES):
        raise HTTPException(status_code=415, detail=f"MIME não permitido: {mime}")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio")
    if len(content) > MEDIA_MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo > {MEDIA_MAX_SIZE // (1024*1024)} MB",
        )

    # Nome único — uuid + extensão original
    import mimetypes
    import uuid as _uuid
    ext = ""
    if "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[1].lower()[:6]
    elif mime != "application/octet-stream":
        ext_guess = mimetypes.guess_extension(mime) or ""
        ext = ext_guess
    safe_name = f"{_uuid.uuid4().hex}{ext}"

    # Upload pro Supabase Storage via REST
    from panel.repos import _supabase_creds
    url, key = _supabase_creds()
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase não configurado")

    import httpx
    upload_url = f"{url}/storage/v1/object/{MEDIA_BUCKET}/{safe_name}"
    public_url = f"{url}/storage/v1/object/public/{MEDIA_BUCKET}/{safe_name}"

    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            upload_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": mime,
                "x-upsert": "true",
            },
            content=content,
        )
        if not r.is_success:
            raise HTTPException(
                status_code=r.status_code,
                detail=f"Upload falhou: {r.text[:300]}",
            )

    await _audit(
        user, "media.upload", "storage_object", safe_name,
        bucket=MEDIA_BUCKET, original_name=file.filename, size=len(content), mime=mime,
    )

    return {
        "ok": True,
        "url": public_url,
        "filename": safe_name,
        "original_name": file.filename,
        "size": len(content),
        "mime": mime,
    }


# ────────────────────────────────────────────────────────────────────
# Leads — Strategist snapshots por (instance, phone) — Reconquista
# ────────────────────────────────────────────────────────────────────

@admin_router.get("/leads")
async def list_leads(
    user: Annotated[dict[str, Any], Depends(require_admin)],
    limit: int = 200,
    project_id: str | None = None,
    temperatura: str | None = None,
) -> list[dict[str, Any]]:
    """Lista snapshots do strategist por lead. Filtros opcionais: project_id, temperatura."""
    from memory.redis_store import RedisStore
    redis = RedisStore()
    rows = await redis.list_lead_statuses(limit=limit)
    # Filtro multi-tenant
    allowed = user.get("project_ids") or []
    if allowed:
        rows = [r for r in rows if r.get("project_id") in allowed]
    if project_id:
        rows = [r for r in rows if r.get("project_id") == project_id]
    if temperatura:
        rows = [r for r in rows if (r.get("temperatura") or "").upper() == temperatura.upper()]
    return rows


@admin_router.delete("/leads/{instance}/{phone}/killswitch")
async def clear_lead_killswitch(
    instance: str,
    phone: str,
    user: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    """Admin override: limpa killswitch + zera attempts (volta a tentar contato)."""
    from memory.redis_store import RedisStore
    redis = RedisStore()
    await redis.reset_followup_attempts(instance, phone)
    status = await redis.get_lead_status(instance, phone)
    if status:
        status["killswitch_permanent"] = False
        status["attempts_made"] = 0
        await redis.set_lead_status(instance, phone, status)
    await _audit(user, "lead.killswitch_cleared", "lead", f"{instance}/{phone}")
    return {"ok": True, "instance": instance, "phone": phone}
