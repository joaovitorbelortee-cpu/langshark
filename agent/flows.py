"""
Catálogo de fluxos pré-cadastrados (sequências text/image/video/document/audio).

Fonte: tabela Supabase `flows` (criada via painel admin /admin/flows).
Cache local TTL 60s — bot pega fluxos novos rapidamente sem hammer no Supabase.

Schema do passo (compat com bot antigo):
  {"type": "text", "content": "..."}
  {"type": "image", "url": "https://...", "caption": "..."}
  {"type": "video", "url": "https://...", "caption": "..."}
  {"type": "audio", "url": "https://..."}
  {"type": "document", "url": "https://...", "fileName": "...", "caption": "..."}

Fallback: FLOW_REGISTRY in-memory (dev/teste OU se Supabase offline).
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass
class Flow:
    name: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""


# Registry in-memory (fallback dev/teste). Produção usa Supabase via cache.
FLOW_REGISTRY: dict[str, dict[str, Flow]] = {}

# Cache do Supabase: {project_id: ({name_lower: Flow}, expires_at)}
_SUPA_CACHE: dict[str, tuple[dict[str, Flow], float]] = {}
_SUPA_CACHE_TTL = 60.0  # 60s — flux novo aparece em <1min sem reiniciar
_SUPA_CACHE_MAX_PROJECTS = 100  # Hard cap pra evitar growth multi-tenant


def register_flow(project_id: str, flow: Flow) -> None:
    FLOW_REGISTRY.setdefault(project_id, {})[flow.name.lower()] = flow


def _supabase_fetch_flows(project_id: str) -> dict[str, Flow]:
    """Busca fluxos ATIVOS do Supabase. Retorna dict {name_lower: Flow}.
    Sync (httpx sync client) — get_flow é chamado de nós sync no graph."""
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = (os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY") or "").strip()
    if not url or not key:
        return {}

    try:
        with httpx.Client(timeout=4.0) as c:
            r = c.get(
                f"{url}/rest/v1/flows",
                params={
                    "select": "*",
                    "project_id": f"eq.{project_id}",
                    "enabled": "eq.true",
                },
                headers={
                    # apikey é suficiente — Supabase REST aceita esse header
                    # sem precisar duplicar como Authorization Bearer
                    "apikey": key,
                },
            )
            if not r.is_success:
                log.warning("[flows] Supabase HTTP %d: %s", r.status_code, r.text[:120])
                return {}
            rows = r.json() or []
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("[flows] Supabase fetch erro: %s", exc)
        return {}

    out: dict[str, Flow] = {}
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        steps = row.get("steps") or []
        if not isinstance(steps, list):
            continue
        out[name.lower()] = Flow(
            name=name,
            steps=steps,
            description=(row.get("description") or "").strip(),
        )
    return out


def _get_supa_flows(project_id: str) -> dict[str, Flow]:
    """Acessa cache; refetch se expirado. Sempre retorna dict (vazio se erro)."""
    pid = project_id or "padrao"
    cached = _SUPA_CACHE.get(pid)
    now = time.time()
    if cached and cached[1] > now:
        return cached[0]
    fresh = _supabase_fetch_flows(pid)
    if fresh or not cached:
        # LRU eviction: se cache lotado, remove entry mais antiga
        if len(_SUPA_CACHE) >= _SUPA_CACHE_MAX_PROJECTS and pid not in _SUPA_CACHE:
            oldest_key = min(_SUPA_CACHE, key=lambda k: _SUPA_CACHE[k][1])
            _SUPA_CACHE.pop(oldest_key, None)
        _SUPA_CACHE[pid] = (fresh, now + _SUPA_CACHE_TTL)
        return fresh
    # Fetch falhou mas tem cache stale — usa stale (gracefulness)
    return cached[0]


def invalidate_flows_cache(project_id: str | None = None) -> None:
    """Force refresh do cache. Chame após CRUD via painel pra ver mudança imediata."""
    if project_id:
        _SUPA_CACHE.pop(project_id, None)
    else:
        _SUPA_CACHE.clear()


def get_flow(project_id: str, name: str) -> Flow | None:
    """Busca fluxo em Supabase (cache 60s) → fallback FLOW_REGISTRY in-memory."""
    pid = project_id or "padrao"
    name_lower = (name or "").strip().lower()
    if not name_lower:
        return None
    # 1ª fonte: Supabase
    supa = _get_supa_flows(pid)
    if name_lower in supa:
        return supa[name_lower]
    # 2ª: registry in-memory (dev/teste)
    return FLOW_REGISTRY.get(pid, {}).get(name_lower)


def list_flows(project_id: str) -> list[Flow]:
    """Lista todos fluxos ativos do projeto (Supabase + registry)."""
    pid = project_id or "padrao"
    supa = _get_supa_flows(pid)
    out = dict(supa)  # Supabase tem prioridade
    for name_lower, flow in FLOW_REGISTRY.get(pid, {}).items():
        out.setdefault(name_lower, flow)
    return list(out.values())


# Regex de detecção (compat com sintaxe antiga [FLOW: <nome>])
_FLOW_TAG = re.compile(r"\[\s*FLOW\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)


def parse_flow_tag(text: str) -> tuple[str | None, str]:
    """Extrai a tag [FLOW: nome] e devolve (flow_name, texto_limpo)."""
    m = _FLOW_TAG.search(text or "")
    if not m:
        return None, text
    name = m.group(1).strip()
    cleaned = _FLOW_TAG.sub("", text).strip()
    return name, cleaned


def flows_prompt_block(project_id: str) -> str:
    """Bloco descritivo dos fluxos disponíveis para injetar no system prompt."""
    flows = list_flows(project_id)
    if not flows:
        return ""
    lines = [
        "<fluxos_cadastrados>",
        "Você pode acionar um fluxo pré-gravado emitindo a tag [FLOW: nome] no FINAL da resposta.",
        "O sistema enviará a sequência cadastrada e ignorará o texto da resposta atual.",
        "Fluxos disponíveis:",
    ]
    for f in flows:
        desc = f" — {f.description}" if f.description else ""
        lines.append(f"- {f.name}{desc}")
    lines.append("</fluxos_cadastrados>")
    return "\n".join(lines)
