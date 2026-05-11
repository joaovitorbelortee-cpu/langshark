"""
Sincroniza ai_models_catalog com a lista canonica do código.

Idempotente — pode rodar quantas vezes quiser, apenas faz UPSERT.

Uso:
  cd D:/claude code/bot-vendas
  python scripts/sync_models_catalog.py

Requer SUPABASE_URL + SUPABASE_SERVICE_KEY no .env.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv


# Catálogo canônico — fonte única de verdade. Adicione novos modelos aqui.
MODELS: list[dict[str, Any]] = [
    # ─── OpenAI GPT-4 line (legacy mas ainda usado) ───
    {"model_id": "openai/gpt-4o-mini",   "provider": "openai", "display_name": "GPT-4o mini",
     "input_price": 0.15, "output_price": 0.60, "tier": "budget",   "supports_vision": True,  "sort_order": 10},
    {"model_id": "openai/gpt-4o",        "provider": "openai", "display_name": "GPT-4o",
     "input_price": 2.50, "output_price": 10.00, "tier": "premium", "supports_vision": True,  "sort_order": 20},

    # ─── OpenAI GPT-5 line ───
    {"model_id": "openai/gpt-5-nano",    "provider": "openai", "display_name": "GPT-5 nano",
     "input_price": 0.05, "output_price": 0.40,  "tier": "budget",   "supports_vision": False, "sort_order": 5},
    {"model_id": "openai/gpt-5-mini",    "provider": "openai", "display_name": "GPT-5 mini",
     "input_price": 0.25, "output_price": 2.00,  "tier": "standard", "supports_vision": True,  "sort_order": 6},
    {"model_id": "openai/gpt-5",         "provider": "openai", "display_name": "GPT-5",
     "input_price": 1.25, "output_price": 10.00, "tier": "premium",  "supports_vision": True,  "sort_order": 7},

    # ─── OpenAI GPT-5.4 (Mar 2026) ───
    {"model_id": "openai/gpt-5.4-nano",  "provider": "openai", "display_name": "GPT-5.4 nano",
     "input_price": 0.20, "output_price": 1.25,  "tier": "budget",   "supports_vision": False, "sort_order": 8},
    {"model_id": "openai/gpt-5.4-mini",  "provider": "openai", "display_name": "GPT-5.4 mini",
     "input_price": 0.75, "output_price": 4.50,  "tier": "standard", "supports_vision": True,  "sort_order": 9},

    # ─── OpenAI GPT-5.5 flagship (Abr 2026) ───
    {"model_id": "openai/gpt-5.5",       "provider": "openai", "display_name": "GPT-5.5",
     "input_price": 5.00, "output_price": 30.00, "tier": "premium",  "supports_vision": True,  "sort_order": 11},

    # ─── Anthropic ───
    {"model_id": "anthropic/claude-haiku-4.5",  "provider": "anthropic", "display_name": "Claude Haiku 4.5",
     "input_price": 1.00, "output_price": 5.00,  "tier": "standard", "supports_vision": True,  "sort_order": 30},
    {"model_id": "anthropic/claude-sonnet-4.5", "provider": "anthropic", "display_name": "Claude Sonnet 4.5",
     "input_price": 3.00, "output_price": 15.00, "tier": "premium",  "supports_vision": True,  "sort_order": 40},

    # ─── Google ───
    {"model_id": "google/gemini-2.5-flash", "provider": "google", "display_name": "Gemini 2.5 Flash",
     "input_price": 0.075, "output_price": 0.30, "tier": "budget",   "supports_vision": True,  "sort_order": 50},
    {"model_id": "google/gemini-2.5-pro",   "provider": "google", "display_name": "Gemini 2.5 Pro",
     "input_price": 1.25, "output_price": 5.00,  "tier": "premium",  "supports_vision": True,  "sort_order": 60},

    # ─── DeepSeek ───
    {"model_id": "deepseek/deepseek-chat", "provider": "deepseek", "display_name": "DeepSeek V3",
     "input_price": 0.27, "output_price": 1.10,  "tier": "budget",   "supports_vision": False, "sort_order": 70},

    # ─── Free ───
    {"model_id": "google/gemma-3-27b-it:free", "provider": "google", "display_name": "Gemma 3 27B (free)",
     "input_price": 0.00, "output_price": 0.00,  "tier": "free",     "supports_vision": True,  "sort_order": 80},
]


def main() -> int:
    load_dotenv()
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
    if not url or not key:
        print("[x] SUPABASE_URL / SUPABASE_SERVICE_KEY ausentes no .env", file=sys.stderr)
        return 1

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }

    print(f"[*] Sincronizando {len(MODELS)} modelos contra {url}")
    with httpx.Client(timeout=15.0) as c:
        # Upsert em batch — Prefer header faz INSERT...ON CONFLICT DO UPDATE
        r = c.post(
            f"{url}/rest/v1/ai_models_catalog",
            headers=headers,
            json=MODELS,
        )
        if not r.is_success:
            print(f"[x] HTTP {r.status_code}: {r.text[:400]}", file=sys.stderr)
            return 1
        rows = r.json() if r.content else []
        print(f"[ok] {len(rows)} rows upserted")
        for row in sorted(rows, key=lambda x: x.get("sort_order", 999)):
            print(f"  - [{row.get('tier'):8s}] {row.get('model_id'):40s} "
                  f"in=${row.get('input_price')}/M out=${row.get('output_price')}/M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
