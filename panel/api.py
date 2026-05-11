"""
Sub-router do painel admin — endpoints JSON sob /api/admin/*.

F1: GETs read-only.
F2: + auth (login/logout/me).
F3-F5: PATCH/POST/DELETE.
"""
from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Response

from agent.tools import EvolutionClient
from panel.auth import (
    authenticate,
    clear_session_cookie,
    create_token,
    require_admin,
    set_session_cookie,
)
from panel.repos import AIModelsCatalogRepo, ProjectConfigRepo


admin_router = APIRouter()
_project_repo = ProjectConfigRepo()
_models_repo = AIModelsCatalogRepo()
_evo = EvolutionClient()


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

@admin_router.get("/projects")
async def list_projects() -> list[dict[str, Any]]:
    rows = await _project_repo.list()
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
async def get_project(project_id: str) -> dict[str, Any]:
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
    body: dict = Body(...),
    user: Annotated[dict[str, Any], Depends(require_admin)] = None,  # type: ignore
) -> dict[str, Any]:
    valid_keys = {"company_info", "prices", "parameters", "priority_situations", "knowledge_base"}
    if section_key not in valid_keys:
        raise HTTPException(status_code=400, detail=f"section_key invalido. validos: {valid_keys}")
    content = body.get("content") or ""
    if len(content) > 7000:
        raise HTTPException(status_code=400, detail="Conteudo > 7000 chars")
    cfg = await _project_repo.patch_section(project_id, section_key, content)
    _invalidate_cache(project_id)
    return {"ok": True, "section": section_key, "size": len(content), "config": cfg}


@admin_router.patch("/projects/{project_id}/config")
async def patch_config(
    project_id: str,
    body: dict = Body(...),
    user: Annotated[dict[str, Any], Depends(require_admin)] = None,  # type: ignore
) -> dict[str, Any]:
    allowed = {"agent_name", "ai_model", "ai_temperature", "ai_max_tokens", "is_active", "display_name"}
    payload = {k: v for k, v in body.items() if k in allowed}
    if not payload:
        raise HTTPException(status_code=400, detail="Nenhum campo valido pra atualizar")
    cfg = await _project_repo.patch(project_id, payload)
    _invalidate_cache(project_id)
    return {"ok": True, "updated": list(payload.keys()), "config": cfg}


# ────────────────────────────────────────────────────────────────────
# AI Models catalog
# ────────────────────────────────────────────────────────────────────

@admin_router.get("/models")
async def list_models() -> list[dict[str, Any]]:
    return await _models_repo.list(only_active=True)


# ────────────────────────────────────────────────────────────────────
# Instances (proxy Evolution + binding instance_projects)
# ────────────────────────────────────────────────────────────────────

@admin_router.get("/instances")
async def list_instances() -> list[dict[str, Any]]:
    """
    Lista instâncias da Evolution API + status de conexão.
    F1: read-only. POST/DELETE em F4.
    """
    base_url = os.getenv("EVOLUTION_API_URL", "").rstrip("/")
    api_key = os.getenv("EVOLUTION_API_KEY", "")
    if not base_url or not api_key:
        return []

    import httpx
    async with httpx.AsyncClient(timeout=8.0) as c:
        try:
            r = await c.get(
                f"{base_url}/instance/fetchInstances",
                headers={"apikey": api_key},
            )
            r.raise_for_status()
            payload = r.json() or []
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Evolution unreachable: {exc}")

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
    return out
