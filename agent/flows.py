"""
Catálogo de fluxos pré-cadastrados (sequências text/image/video/document/audio).

No bot antigo isso vinha do Supabase. Aqui exponho duas fontes:
  - In-memory registry (FLOW_REGISTRY) — útil pra dev/teste.
  - get_flow(project_id, name) → retorna o fluxo. Plugue aqui o adapter Supabase
    quando integrar com a tabela `flows` antiga.

Schema do passo (compat com bot antigo):
  {"type": "text", "content": "..."}
  {"type": "image", "url": "https://...", "caption": "..."}
  {"type": "video", "url": "https://...", "caption": "..."}
  {"type": "audio", "url": "https://..."}
  {"type": "document", "url": "https://...", "fileName": "...", "caption": "..."}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Flow:
    name: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""


# Registry simples por projeto. Use register_flow() para popular no startup,
# ou substitua get_flow() por um adapter Supabase.
FLOW_REGISTRY: dict[str, dict[str, Flow]] = {}


def register_flow(project_id: str, flow: Flow) -> None:
    FLOW_REGISTRY.setdefault(project_id, {})[flow.name.lower()] = flow


def get_flow(project_id: str, name: str) -> Flow | None:
    """Override-me com adapter Supabase em produção."""
    return FLOW_REGISTRY.get(project_id, {}).get(name.lower())


def list_flows(project_id: str) -> list[Flow]:
    return list(FLOW_REGISTRY.get(project_id, {}).values())


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
