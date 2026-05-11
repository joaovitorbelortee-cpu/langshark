"""
Views Jinja2 do painel admin.

Páginas:
  GET /admin/login      — form login
  GET /admin            — dashboard (protegido)
  GET /admin/agent      — cérebro do agente (F3)
  GET /admin/instances  — F4
  GET /admin/flows      — F4
  ...
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from panel.auth import (
    authenticate,
    clear_session_cookie,
    create_token,
    get_current_admin_optional,
    require_admin,
    set_session_cookie,
)
from panel.repos import AIModelsCatalogRepo, ProjectConfigRepo


_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

views_router = APIRouter()
_project_repo = ProjectConfigRepo()
_models_repo = AIModelsCatalogRepo()


# ────────────────────────────────────────────────────────────────────
# Login / logout
# ────────────────────────────────────────────────────────────────────

@views_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = await get_current_admin_optional(request)
    if user:
        return RedirectResponse("/admin", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@views_router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    user = await authenticate(email, password)
    if not user:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Email ou senha invalidos"},
            status_code=401,
        )
    token = create_token(user)
    response = RedirectResponse("/admin", status_code=302)
    set_session_cookie(response, token)
    return response


@views_router.get("/logout")
async def logout():
    response = RedirectResponse("/admin/login", status_code=302)
    clear_session_cookie(response)
    return response


# ────────────────────────────────────────────────────────────────────
# Dashboard (protegido)
# ────────────────────────────────────────────────────────────────────

def _pick_current_project(projects: list[dict[str, Any]], requested: str | None) -> dict[str, Any] | None:
    """Resolve projeto atual: query string > 'padrao' > primeiro com agent_name > primeiro."""
    if not projects:
        return None
    if requested:
        for p in projects:
            if p["project_id"] == requested:
                return p
    for p in projects:
        if p["project_id"] == "padrao":
            return p
    for p in projects:
        if p.get("agent_name"):
            return p
    return projects[0]


@views_router.get("", response_class=HTMLResponse)
@views_router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"))
    kpis = {
        "instances_online": 1,
        "messages_24h": 0,
        "conversions_24h": 0,
        "active_leads": 0,
    }
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "projects": projects,
            "current_project": current,
            "kpis": kpis,
            "active_section": "dashboard",
        },
    )


# ────────────────────────────────────────────────────────────────────
# Stub das outras páginas (placeholder F3-F4)
# ────────────────────────────────────────────────────────────────────

@views_router.get("/agent", response_class=HTMLResponse)
async def agent_page(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"))
    pid = current["project_id"] if current else "padrao"
    cfg = await _project_repo.fetch(pid) or {}
    models = await _models_repo.list()
    return templates.TemplateResponse(
        request,
        "agent.html",
        {
            "user": user,
            "projects": projects,
            "current_project": current or cfg,
            "config": cfg,
            "sections": (cfg.get("brain_sections") or {}),
            "models": models,
            "active_section": "agent",
        },
    )


@views_router.get("/{section}", response_class=HTMLResponse)
async def stub_section(
    section: str,
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    valid = {"instances", "flows", "knowledge", "recovery", "settings"}
    if section not in valid:
        raise HTTPException(404, "Pagina nao encontrada")
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"))
    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "user": user,
            "projects": projects,
            "current_project": current,
            "active_section": section,
            "section_title": section.capitalize(),
        },
    )
