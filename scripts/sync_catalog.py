"""
Sync catálogo de produtos do Supabase pro ChromaDB local.

Lê tabela `products` (schema na migration `0001_products.sql`) e indexa no Chroma.

Uso:
    python -m scripts.sync_catalog                  # sync default project
    python -m scripts.sync_catalog --project foo    # sync project específico
    python -m scripts.sync_catalog --all            # sync todos os projects

Variáveis necessárias: SUPABASE_URL, SUPABASE_SERVICE_KEY (ou ANON_KEY).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

from rag.catalog import CatalogRAG, Product

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("sync_catalog")


async def fetch_products(project_id: str | None = None) -> list[dict[str, Any]]:
    """GET /rest/v1/products. Filtra por project_id se fornecido."""
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
    if not url or not key:
        log.error("SUPABASE_URL e SUPABASE_SERVICE_KEY são obrigatórios")
        sys.exit(1)

    params: dict[str, str] = {
        "select": "id,project_id,name,description,price,metadata",
        "limit": "1000",
    }
    if project_id:
        params["project_id"] = f"eq.{project_id}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            f"{url}/rest/v1/products",
            params=params,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
        )
        r.raise_for_status()
        return r.json() or []


def index_products(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Agrupa por project_id e upserta em coleções separadas."""
    rag = CatalogRAG()
    by_project: dict[str, list[Product]] = {}
    for row in rows:
        pid = row.get("project_id") or "padrao"
        price = row.get("price")
        by_project.setdefault(pid, []).append(
            Product(
                id=str(row["id"]),
                name=row.get("name", ""),
                description=row.get("description", ""),
                price=float(price) if price is not None else None,
                metadata=row.get("metadata") or {},
            )
        )

    counts: dict[str, int] = {}
    for pid, products in by_project.items():
        n = rag.upsert_products(pid, products)
        counts[pid] = n
        log.info("[sync] %s: %d produtos indexados", pid, n)
    return counts


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("DEFAULT_PROJECT_ID", "padrao"))
    parser.add_argument("--all", action="store_true", help="Sync todos os projetos")
    args = parser.parse_args()

    pid_filter = None if args.all else args.project
    rows = await fetch_products(pid_filter)
    log.info("[sync] %d produtos lidos do Supabase", len(rows))

    if not rows:
        log.warning("Nenhum produto encontrado.")
        return

    counts = index_products(rows)
    total = sum(counts.values())
    log.info("[sync] ✓ Total: %d produtos em %d projetos", total, len(counts))


if __name__ == "__main__":
    asyncio.run(main())
