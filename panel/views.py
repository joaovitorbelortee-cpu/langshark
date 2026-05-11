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

def _pick_current_project(
    projects: list[dict[str, Any]],
    requested: str | None,
    user: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Resolve projeto atual: query string (se autorizado) > 'padrao' > 1º com agent_name > 1º.

    Multi-tenant: se user tem project_ids restrito, só seleciona dentre os autorizados.
    """
    allowed = (user or {}).get("project_ids") or []
    pool = [p for p in projects if (not allowed) or (p["project_id"] in allowed)]
    if not pool:
        return None
    if requested:
        for p in pool:
            if p["project_id"] == requested:
                return p
        # query string aponta pra projeto fora do escopo — ignora silenciosamente
    for p in pool:
        if p["project_id"] == "padrao":
            return p
    for p in pool:
        if p.get("agent_name"):
            return p
    return pool[0]


@views_router.get("", response_class=HTMLResponse)
@views_router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"), user)
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
    current = _pick_current_project(projects, request.query_params.get("project_id"), user)
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


@views_router.get("/instances", response_class=HTMLResponse)
async def instances_page(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"), user)
    return templates.TemplateResponse(
        request,
        "instances.html",
        {
            "user": user, "projects": projects, "current_project": current,
            "active_section": "instances", "page_title": "Instâncias",
        },
    )


@views_router.get("/flows", response_class=HTMLResponse)
async def flows_page(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"), user)
    return templates.TemplateResponse(
        request,
        "flows.html",
        {
            "user": user, "projects": projects, "current_project": current,
            "active_section": "flows", "page_title": "Fluxos Inteligentes",
        },
    )


@views_router.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"), user)
    return templates.TemplateResponse(
        request,
        "knowledge.html",
        {
            "user": user, "projects": projects, "current_project": current,
            "active_section": "knowledge", "page_title": "Base de Conhecimento",
        },
    )


@views_router.get("/recovery", response_class=HTMLResponse)
async def recovery_page(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"), user)
    cfg = await _project_repo.fetch(current["project_id"]) if current else {}
    return templates.TemplateResponse(
        request,
        "recovery.html",
        {
            "user": user, "projects": projects, "current_project": current,
            "config": cfg or {},
            "active_section": "recovery", "page_title": "Reconquista",
        },
    )


@views_router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    user: Annotated[dict[str, Any], Depends(require_admin)],
):
    import os
    projects = await _project_repo.list()
    current = _pick_current_project(projects, request.query_params.get("project_id"), user)
    env_status = {
        "webhook_url":  f"https://{os.getenv('PUBLIC_BASE_URL', 'bot-vendas-production-0be5.up.railway.app').replace('https://','').rstrip('/')}/webhook/evolution",
        "evolution":    bool(os.getenv("EVOLUTION_API_URL")),
        "redis":        bool(os.getenv("UPSTASH_REDIS_REST_URL")),
        "qstash":       bool(os.getenv("QSTASH_TOKEN")),
        "openrouter":   bool(os.getenv("OPENROUTER_API_KEY")),
        "supabase":     bool(os.getenv("SUPABASE_URL")),
        "postgres":     bool(os.getenv("POSTGRES_URL")),
    }
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "user": user, "projects": projects, "current_project": current,
            "env_status": env_status,
            "active_section": "settings", "page_title": "Configurações",
        },
    )
