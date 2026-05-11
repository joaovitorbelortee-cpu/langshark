"""
Particiona o SALES_SYSTEM hardcoded em 5 seções editáveis e gravável no Supabase.

Uso:
    python -m scripts.seed_brain                # projeto padrao
    python -m scripts.seed_brain --project foo

Lê SALES_SYSTEM de agent.nodes (fonte única de verdade), divide em seções
baseadas nos cabeçalhos conhecidos do prompt Game Pass, e UPDATE no Supabase.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv

from agent.nodes import SALES_SYSTEM

load_dotenv()


SECTION_DEFS = [
    # (key, title, icon, [start_markers — primeiro match marca início])
    ("company_info", "Informacoes da Empresa", "building",
     ["QUEM VOCE E"]),
    ("prices", "Precos e Valores", "dollar",
     ["PRECOS E LINKS"]),
    ("parameters", "Parametros", "settings",
     ["REGRAS GERAIS:"]),
    ("priority_situations", "Situacoes Prioritarias", "alert",
     ["PILARES DE PERSUASAO"]),
    ("knowledge_base", "Base de Conhecimento", "book",
     []),  # vazio — admin preenche depois
]

# Stop markers — onde cortar a próxima seção
STOP_AT = "<regras_estritas>"   # tudo dali em diante vira footer auto


def split_prompt(prompt: str) -> dict[str, str]:
    """Divide SALES_SYSTEM por marcadores conhecidos."""
    text = prompt.split(STOP_AT, 1)[0].strip()  # corta footer técnico
    sections: dict[str, str] = {key: "" for key, *_ in SECTION_DEFS}

    # Encontra posições de cada marcador
    positions: list[tuple[int, str]] = []
    for key, _title, _icon, markers in SECTION_DEFS:
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0:
                positions.append((idx, key))
                break

    positions.sort(key=lambda p: p[0])

    # Slice do prompt em pedaços por posição
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[key] = text[start:end].strip()

    return sections


async def upsert_brain(project_id: str, sections: dict[str, str]) -> None:
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
    if not url or not key:
        print("ERRO: SUPABASE_URL e SUPABASE_SERVICE_KEY são obrigatórios", file=sys.stderr)
        sys.exit(1)

    brain_payload = {
        key: {
            "content": sections.get(key, ""),
            "max_chars": 7000,
            "icon": icon,
            "title": title,
        }
        for key, title, icon, _ in SECTION_DEFS
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Upsert via PATCH (project_config row já existe via migration)
        r = await client.patch(
            f"{url}/rest/v1/project_config",
            params={"project_id": f"eq.{project_id}"},
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json={"brain_sections": brain_payload},
        )
        if r.status_code >= 400:
            print(f"ERRO {r.status_code}: {r.text}", file=sys.stderr)
            sys.exit(1)
        data = r.json()
        if not data:
            # Row não existia — cria
            r2 = await client.post(
                f"{url}/rest/v1/project_config",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                },
                json={
                    "project_id": project_id,
                    "display_name": "Meu Primeiro Projeto",
                    "agent_name": "Joao",
                    "brain_sections": brain_payload,
                },
            )
            r2.raise_for_status()

    for key in brain_payload:
        size = len(brain_payload[key]["content"])
        status = "Preenchido" if size else "Vazio"
        print(f"  [{status:10s}] {key:20s} {size:>5d} chars")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="padrao")
    args = parser.parse_args()

    sections = split_prompt(SALES_SYSTEM)
    print(f"\n[seed_brain] Particionando SALES_SYSTEM em 5 secoes para projeto '{args.project}':\n")
    await upsert_brain(args.project, sections)
    print(f"\n[seed_brain] OK\n")


if __name__ == "__main__":
    asyncio.run(main())
