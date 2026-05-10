"""
RAG do catálogo via Supabase (substitui ChromaDB local em produção).

Vantagens vs ChromaDB local em container:
  - Persistente (não perde em redeploy/restart)
  - Compartilhado entre instâncias do app
  - Update via SQL/dashboard sem redeploy

Estratégia: filtro `ilike` por nome+descrição com ranking simples por overlap
de palavras. Suficiente para catálogos de até ~500 produtos. Pra escala maior,
adicionar índice GIN com pg_trgm + similarity().
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)


_STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "ou", "para",
    "por", "com", "sem", "no", "na", "nos", "nas", "um", "uma", "uns", "umas",
    "que", "se", "em", "ao", "à", "às", "aos", "este", "esta", "isso", "isto",
    "qual", "quais", "como", "tem", "ter", "ser", "é",
}


def _tokenize(text: str) -> list[str]:
    """Normaliza string em palavras minúsculas, sem acentos triviais."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [t for t in text.split() if len(t) > 2 and t not in _STOPWORDS]
    return tokens


class SupabaseRAG:
    """
    Adapter RAG sobre Supabase.

    Compatível com a interface de `rag.catalog.CatalogRAG` (search + format_context).
    """

    def __init__(
        self,
        url: str | None = None,
        service_key: str | None = None,
        timeout: float = 5.0,
    ):
        self.url = (url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.key = (
            service_key
            or os.getenv("SUPABASE_SERVICE_KEY", "")
            or os.getenv("SUPABASE_ANON_KEY", "")
        )
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
        }

    async def _fetch_products(self, project_id: str) -> list[dict[str, Any]]:
        """GET /rest/v1/products filtrado por project_id."""
        if not self.enabled:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                r = await client.get(
                    f"{self.url}/rest/v1/products",
                    params={
                        "select": "id,name,description,price",
                        "project_id": f"eq.{project_id}",
                        "limit": "200",
                    },
                    headers=self._headers(),
                )
                r.raise_for_status()
                return r.json() or []
            except Exception as exc:  # noqa: BLE001
                log.warning("[supabase_rag] fetch failed: %s", exc)
                return []

    async def asearch(
        self,
        project_id: str,
        query: str,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Async search: busca produtos do project_id, ranqueia por overlap
        de tokens com a query, devolve top_k.
        """
        products = await self._fetch_products(project_id)
        if not products:
            return []

        q_tokens = set(_tokenize(query))
        if not q_tokens:
            # Sem tokens úteis na query: devolve primeiros N como prova social.
            return [
                {
                    "id": p["id"],
                    "name": p.get("name", ""),
                    "description": p.get("description", ""),
                    "price": float(p["price"]) if p.get("price") is not None else None,
                    "score": 0.0,
                }
                for p in products[:top_k]
            ]

        scored: list[tuple[float, dict[str, Any]]] = []
        for p in products:
            haystack = f"{p.get('name', '')} {p.get('description', '')}"
            doc_tokens = set(_tokenize(haystack))
            if not doc_tokens:
                continue
            overlap = len(q_tokens & doc_tokens)
            jaccard = overlap / len(q_tokens | doc_tokens)
            scored.append((jaccard, p))

        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, p in scored[:top_k]:
            if score <= 0:
                continue
            out.append(
                {
                    "id": p["id"],
                    "name": p.get("name", ""),
                    "description": p.get("description", ""),
                    "price": float(p["price"]) if p.get("price") is not None else None,
                    "score": score,
                }
            )
        return out

    def search(self, project_id: str, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Sync wrapper — fallback pra quem precisa de chamada síncrona."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asearch(project_id, query, top_k))

        # Já em loop: usa ensure_future + sync wait via run_until_complete não dá.
        # Caso raro — devolve vazio em vez de bloquear o loop.
        return []

    def format_context(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return ""
        lines = ["<catalogo_relevante>"]
        for h in hits:
            price = f"R$ {h['price']:.2f}" if h.get("price") is not None else "consultar"
            lines.append(f"- {h['name']} ({price}): {h['description']}")
        lines.append("</catalogo_relevante>")
        return "\n".join(lines)
