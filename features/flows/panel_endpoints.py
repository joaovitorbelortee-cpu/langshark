"""
Endpoints FastAPI CRUD pra flows (painel admin).

Plugue em qualquer FastAPI app:
    from features.flows.panel_endpoints import router as flows_admin_router
    app.include_router(flows_admin_router, prefix="/api/admin")

Requer:
- SUPABASE_URL + SUPABASE_SERVICE_KEY env vars
- Sua função de auth (substitua `require_admin` placeholder pelo seu dependency)
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException

from .flows import invalidate_flows_cache


router = APIRouter()


# ─── Placeholder auth — substitua pelo seu dependency ────────────────
async def require_admin() -> dict[str, Any]:
    """Override no seu app com Depends(seu_auth_real)."""
    return {"id": "anon", "project_ids": []}


def _user_can_access(user: dict, project_id: str) -> bool:
    allowed = user.get("project_ids") or []
    return (not allowed) or (project_id in allowed)


# ─── Supabase REST helpers ───────────────────────────────────────────
def _supabase() -> tuple[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
    if not url or not key:
        raise HTTPException(503, "Supabase não configurado (SUPABASE_URL/SERVICE_KEY)")
    return url, key


def _headers() -> dict[str, str]:
    _url, key = _supabase()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# ─── CRUD endpoints ──────────────────────────────────────────────────

@router.get("/flows")
async def list_flows_admin(
    user: dict = Depends(require_admin),
    project_id: str = "padrao",
) -> list[dict[str, Any]]:
    if not _user_can_access(user, project_id):
        raise HTTPException(403, "Forbidden")
    url, _ = _supabase()
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(
            f"{url}/rest/v1/flows",
            params={"select": "*", "project_id": f"eq.{project_id}"},
            headers=_headers(),
        )
        r.raise_for_status()
        return r.json() or []


@router.post("/flows")
async def create_flow(
    body: dict = Body(...),
    user: dict = Depends(require_admin),
) -> dict[str, Any]:
    project_id = body.get("project_id") or "padrao"
    if not _user_can_access(user, project_id):
        raise HTTPException(403, "Forbidden")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name obrigatorio")
    url, _ = _supabase()
    payload = {
        "project_id":  project_id,
        "name":        name,
        "description": body.get("description") or "",
        "steps":       body.get("steps") or [],
        "enabled":     body.get("enabled", True),
    }
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(
            f"{url}/rest/v1/flows",
            headers={**_headers(), "Prefer": "return=representation"},
            json=payload,
        )
        r.raise_for_status()
        rows = r.json() or []
        row = rows[0] if rows else {}
    invalidate_flows_cache(project_id)
    return {"ok": True, "flow": row}


@router.patch("/flows/{flow_id}")
async def patch_flow(
    flow_id: str,
    body: dict = Body(...),
    user: dict = Depends(require_admin),
) -> dict[str, Any]:
    allowed_fields = {"name", "description", "steps", "enabled"}
    payload = {k: v for k, v in body.items() if k in allowed_fields}
    if not payload:
        raise HTTPException(400, "Nenhum campo válido pra atualizar")
    url, _ = _supabase()
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.patch(
            f"{url}/rest/v1/flows",
            params={"id": f"eq.{flow_id}"},
            headers={**_headers(), "Prefer": "return=representation"},
            json=payload,
        )
        r.raise_for_status()
        rows = r.json() or []
        row = rows[0] if rows else {}
    pid = row.get("project_id") if isinstance(row, dict) else None
    invalidate_flows_cache(pid)
    return {"ok": True, "flow": row}


@router.delete("/flows/{flow_id}")
async def delete_flow(
    flow_id: str,
    user: dict = Depends(require_admin),
) -> dict[str, Any]:
    url, _ = _supabase()
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.delete(
            f"{url}/rest/v1/flows",
            params={"id": f"eq.{flow_id}"},
            headers=_headers(),
        )
        r.raise_for_status()
    invalidate_flows_cache(None)  # global invalidate (sem fetch pro project_id)
    return {"ok": True}
