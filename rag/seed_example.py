"""
Script de exemplo para popular o catálogo ChromaDB.

Uso:
    python -m rag.seed_example padrao
"""
from __future__ import annotations

import sys

from rag.catalog import CatalogRAG, Product


def main(project_id: str = "padrao") -> None:
    rag = CatalogRAG()
    products = [
        Product(
            id="plano-basico",
            name="Plano Básico",
            description="Bot WhatsApp com IA, 1 número, até 1000 mensagens/mês.",
            price=97.0,
        ),
        Product(
            id="plano-pro",
            name="Plano Pro",
            description="Bot WhatsApp com IA, 3 números, mensagens ilimitadas, RAG e follow-up automático.",
            price=297.0,
        ),
        Product(
            id="plano-empresa",
            name="Plano Empresa",
            description="Multi-tenant, instâncias ilimitadas, integração custom e suporte dedicado.",
            price=997.0,
        ),
    ]
    n = rag.upsert_products(project_id, products)
    print(f"Indexei {n} produtos no projeto '{project_id}'.")
    print("Teste rápido:", rag.search(project_id, "quero um plano com follow-up automático"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "padrao")
