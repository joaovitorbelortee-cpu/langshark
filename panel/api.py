"""
Sub-router do painel admin — endpoints JSON sob /api/admin/*.

F1: somente GETs read-only (lista projects, project detail, models catalog, instances).
F2-F5: PATCH/POST/DELETE + auth.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException

from agent.tools import EvolutionClient
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
