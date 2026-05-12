"""
Catálogo de fluxos pré-cadastrados (sequências text/image/video/document/audio).

Fontes:
  - Supabase (tabela public.flows) — produção, gerenciado pelo painel admin
  - In-memory registry (FLOW_REGISTRY) — dev/teste, fallback se Supabase offline

Cache 60s nos fetches Supabase pra não martelar REST a cada turno.

Schema do passo (compat com bot antigo):
  {"type": "text", "content": "..."}
  {"type": "image", "url": "https://...", "caption": "..."}
  {"type": "video", "url": "https://...", "caption": "..."}
  {"type": "audio", "url": "https://..."}
  {"type": "document", "url": "https://...", "fileName": "...", "caption": "..."}

Disparo:
  LLM emite tag [FLOW: nome] no final da resposta. parse_flow_tag extrai.
  graph._route_after_reply detecta flow_name no state → flow_executor_node
  envia sequência via Evolution API. supervisor segue normal.
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


# Registry simples por projeto. Fallback se Supabase offline ou pra testes.
FLOW_REGISTRY: dict[str, dict[str, Flow]] = {}


def register_flow(project_id: str, flow: Flow) -> None:
    FLOW_REGISTRY.setdefault(project_id, {})[flow.name.lower()] = flow
    # Invalida cache pra novo flow aparecer no próximo get_flow/list_flows
    # (necessário pra testes + qualquer registro runtime).
    invalidate_flows_cache(project_id)


# ──────────────────────────────────────────────────────────────────────
# Supabase fetch + cache (60s TTL)
# ──────────────────────────────────────────────────────────────────────

_FLOWS_CACHE: dict[str, tuple[list[Flow], float]] = {}
_FLOWS_CACHE_TTL = 60.0
_FLOWS_CACHE_MAX = 100  # LRU eviction multi-tenant


def _supabase_creds() -> tuple[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
    if not key and os.getenv("SUPABASE_ALLOW_ANON") == "1":
        key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    return url, key


def _fetch_flows_supabase(project_id: str) -> list[Flow]:
    """
    Busca flows do Supabase via REST. Síncrono (todos callers são sync).
    Retorna [] se Supabase indisponível ou sem flows.
    """
    url, key = _supabase_creds()
    if not url or not key:
        return []
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(
                f"{url}/rest/v1/flows",
                params={
                    "select": "name,description,steps,enabled",
                    "project_id": f"eq.{project_id}",
                    "enabled": "eq.true",
                },
                headers={"apikey": key, "Accept": "application/json"},
            )
            r.raise_for_status()
            rows = r.json() or []
    except Exception as exc:  # noqa: BLE001
        log.warning("[flows] fetch supabase falhou (%s) — fallback registry", exc)
        return []

    flows: list[Flow] = []
    for row in rows:
        try:
            flows.append(Flow(
                name=str(row.get("name") or "").strip(),
                description=str(row.get("description") or "").strip(),
                steps=list(row.get("steps") or []),
            ))
        except Exception as exc:  # noqa: BLE001
            log.warning("[flows] row inválida ignorada: %s", exc)
            continue
    return [f for f in flows if f.name]


def _cached_flows(project_id: str) -> list[Flow]:
    """Cache 60s. Combina Supabase + REGISTRY (Supabase tem prioridade por nome)."""
    now = time.time()
    entry = _FLOWS_CACHE.get(project_id)
    if entry and entry[1] > now:
        return entry[0]

    # LRU evict
    if len(_FLOWS_CACHE) >= _FLOWS_CACHE_MAX:
        oldest = min(_FLOWS_CACHE, key=lambda k: _FLOWS_CACHE[k][1])
        _FLOWS_CACHE.pop(oldest, None)

    supa = _fetch_flows_supabase(project_id)
    reg = list(FLOW_REGISTRY.get(project_id, {}).values())
    # Merge: Supabase ganha em conflito de nome (case-insensitive)
    seen = {f.name.lower() for f in supa}
    merged = supa + [f for f in reg if f.name.lower() not in seen]

    _FLOWS_CACHE[project_id] = (merged, now + _FLOWS_CACHE_TTL)
    if merged:
        log.info("[flows] %s: carregou %d flow(s) — %s", project_id, len(merged),
                 [f.name for f in merged])
    return merged


def invalidate_flows_cache(project_id: str | None = None) -> None:
    """Limpa cache (chamado pelo painel admin pós-mutation)."""
    if project_id:
        _FLOWS_CACHE.pop(project_id, None)
    else:
        _FLOWS_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────

def get_flow(project_id: str, name: str) -> Flow | None:
    """Match case-insensitive em Supabase + registry (com cache 60s)."""
    if not name:
        return None
    target = name.lower().strip()
    for f in _cached_flows(project_id):
        if f.name.lower() == target:
            return f
    return None


def list_flows(project_id: str) -> list[Flow]:
    """Todos flows do projeto (Supabase + registry, cache 60s)."""
    return _cached_flows(project_id)


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
        "REGRA: só dispare flow quando a situação descrita em 'quando usar' bater CLARAMENTE",
        "com a mensagem do lead. Em dúvida, NÃO dispare — responda normal.",
        "Fluxos disponíveis:",
    ]
    for f in flows:
        desc = f" — quando usar: {f.description}" if f.description else ""
        lines.append(f"- {f.name}{desc}")
    lines.append("</fluxos_cadastrados>")
    return "\n".join(lines)
