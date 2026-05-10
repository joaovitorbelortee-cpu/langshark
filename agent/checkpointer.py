"""
Checkpointer LangGraph com fallback in-memory.

Permite retomada automática de conversas — se o webhook cair no meio do grafo,
ao receber outra mensagem do mesmo telefone, o grafo retoma do último checkpoint.

Hierarquia de escolha:
  1. POSTGRES_URL setado → AsyncPostgresSaver (durável, multi-processo)
  2. Sem POSTGRES_URL → InMemorySaver (dev/teste — não sobrevive a restart)

O thread_id é derivado de (project_id, instance, phone) — uma thread por lead.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

log = logging.getLogger(__name__)


class CheckpointerProvider:
    """
    Wrapper assíncrono que decide o checkpointer em runtime.

    Uso:
        provider = CheckpointerProvider()
        async with provider.context() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            await graph.ainvoke(state, config={"configurable": {"thread_id": ...}})

    Para hot-path do webhook usamos um checkpointer compartilhado por processo
    (provider.shared()), pra evitar criar pool de Postgres a cada request.
    """

    def __init__(self, postgres_url: str | None = None):
        self.postgres_url = postgres_url or os.getenv("POSTGRES_URL", "").strip()
        self._shared: Any | None = None
        self._postgres_ctx = None
        self._postgres_obj: Any | None = None

    @property
    def kind(self) -> str:
        return "postgres" if self.postgres_url else "memory"

    async def shared(self) -> Any:
        """Retorna um checkpointer singleton pro processo."""
        if self._shared is not None:
            return self._shared

        if not self.postgres_url:
            log.info("[checkpointer] usando InMemorySaver (POSTGRES_URL ausente)")
            self._shared = InMemorySaver()
            return self._shared

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            self._postgres_ctx = AsyncPostgresSaver.from_conn_string(self.postgres_url)
            self._postgres_obj = await self._postgres_ctx.__aenter__()
            await self._postgres_obj.setup()
            log.info("[checkpointer] AsyncPostgresSaver pronto (%s)", _mask(self.postgres_url))
            self._shared = self._postgres_obj
            return self._shared
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[checkpointer] falha ao abrir Postgres (%s) — fallback InMemorySaver",
                exc,
            )
            self._shared = InMemorySaver()
            return self._shared

    async def aclose(self) -> None:
        """Fecha o pool de Postgres no shutdown (FastAPI lifespan)."""
        if self._postgres_ctx is not None and self._postgres_obj is not None:
            try:
                await self._postgres_ctx.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                log.warning("[checkpointer] erro ao fechar Postgres: %s", exc)
        self._shared = None
        self._postgres_ctx = None
        self._postgres_obj = None


def _mask(url: str) -> str:
    """Esconde senha em URL postgres."""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def thread_id_for(project_id: str, instance: str, phone: str) -> str:
    """Chave de checkpoint estável por lead."""
    return f"{project_id}:{instance}:{phone}"
