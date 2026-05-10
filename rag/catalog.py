"""
RAG do catálogo de produtos via ChromaDB local.

Substitui o knowledge.txt concatenado do bot antigo por uma busca semântica.
Cada projeto (multi-tenant) tem sua própria coleção: catalog_{project_id}.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings


@dataclass
class Product:
    id: str
    name: str
    description: str
    price: float | None = None
    metadata: dict | None = None


class CatalogRAG:
    """Wrapper sobre ChromaDB persistente para o catálogo multi-tenant."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_DIR", "./chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )

    def _collection_name(self, project_id: str) -> str:
        # ChromaDB exige [a-zA-Z0-9._-], 3-63 chars. Sanitiza projectId.
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in project_id)
        return f"catalog_{safe}"[:63]

    def _collection(self, project_id: str):
        return self.client.get_or_create_collection(
            name=self._collection_name(project_id),
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_products(self, project_id: str, products: list[Product]) -> int:
        """Adiciona/atualiza produtos no catálogo do projeto."""
        if not products:
            return 0
        col = self._collection(project_id)
        col.upsert(
            ids=[p.id for p in products],
            documents=[f"{p.name}\n{p.description}" for p in products],
            metadatas=[
                {
                    "name": p.name,
                    "description": p.description,
                    "price": p.price if p.price is not None else -1,
                    **(p.metadata or {}),
                }
                for p in products
            ],
        )
        return len(products)

    def search(
        self,
        project_id: str,
        query: str,
        top_k: int = 4,
    ) -> list[dict]:
        """Busca produtos relevantes para a mensagem do cliente."""
        if not query.strip():
            return []

        col = self._collection(project_id)
        try:
            res = col.query(query_texts=[query], n_results=top_k)
        except Exception:
            return []

        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[dict] = []
        for i, _id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            distance = dists[i] if i < len(dists) else 1.0
            price = meta.get("price")
            out.append(
                {
                    "id": _id,
                    "name": meta.get("name", ""),
                    "description": meta.get("description", ""),
                    "price": None if price in (None, -1) else float(price),
                    "score": 1.0 - float(distance),  # cosine: menor = mais próximo
                }
            )
        return out

    def format_context(self, hits: list[dict]) -> str:
        """Formata os hits para injetar no prompt da IA."""
        if not hits:
            return ""
        lines = ["<catalogo_relevante>"]
        for h in hits:
            price = f"R$ {h['price']:.2f}" if h.get("price") is not None else "consultar"
            lines.append(f"- {h['name']} ({price}): {h['description']}")
        lines.append("</catalogo_relevante>")
        return "\n".join(lines)
