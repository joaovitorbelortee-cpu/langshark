"""
Checkpointer LangGraph com fallback in-memory.

Permite retomada automática de conversas — se o webhook cair no meio do grafo,
ao receber outra mensagem do mesmo telefone, o grafo retoma do último checkpoint.

Hierarquia de escolha (1ª que funcionar):
  1. REDIS_URL setado → AsyncRedisSaver (preferido: 1 backend só, vector search opcional)
  2. POSTGRES_URL setado → AsyncPostgresSaver (fallback durável)
  3. InMemorySaver (dev/teste — não sobrevive a restart)

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
        cp = await provider.shared()
        graph = build_graph(checkpointer=cp)

    Para hot-path do webhook usamos um checkpointer compartilhado por processo
    (provider.shared()), pra evitar criar pool a cada request.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        postgres_url: str | None = None,
    ):
        self.redis_url = (redis_url or os.getenv("REDIS_URL", "")).strip()
        self.postgres_url = (postgres_url or os.getenv("POSTGRES_URL", "")).strip()
        self._shared: Any | None = None
        # Postgres ctx
        self._postgres_ctx = None
        self._postgres_obj: Any | None = None
        # Redis ctx
        self._redis_ctx = None
        self._redis_obj: Any | None = None
        self._kind: str = "uninit"

    @property
    def kind(self) -> str:
        return self._kind

    async def shared(self) -> Any:
        """Retorna um checkpointer singleton pro processo."""
        if self._shared is not None:
            return self._shared

        # 1ª escolha: Redis nativo
        if self.redis_url:
            try:
                from langgraph.checkpoint.redis.aio import AsyncRedisSaver

                self._redis_ctx = AsyncRedisSaver.from_conn_string(self.redis_url)
                self._redis_obj = await self._redis_ctx.__aenter__()
                # asetup cria índices RedisJSON/RediSearch necessários
                try:
                    await self._redis_obj.asetup()
                except Exception as setup_exc:  # noqa: BLE001
                    log.warning("[checkpointer] asetup Redis falhou (idx talvez já existe): %s", setup_exc)
                self._shared = self._redis_obj
                self._kind = "redis"
                log.info("[checkpointer] AsyncRedisSaver pronto (%s)", _mask(self.redis_url))
                return self._shared
            except ImportError as exc:
                log.warning("[checkpointer] langgraph-checkpoint-redis não instalado: %s", exc)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[checkpointer] falha ao abrir Redis (%s) — tenta Postgres fallback",
                    exc,
                )

        # 2ª escolha: Postgres
        if self.postgres_url:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                self._postgres_ctx = AsyncPostgresSaver.from_conn_string(self.postgres_url)
                self._postgres_obj = await self._postgres_ctx.__aenter__()
                await self._postgres_obj.setup()
                self._shared = self._postgres_obj
                self._kind = "postgres"
                log.info("[checkpointer] AsyncPostgresSaver pronto (%s)", _mask(self.postgres_url))
                return self._shared
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[checkpointer] falha ao abrir Postgres (%s) — fallback InMemorySaver",
                    exc,
                )

        # Último recurso: in-memory
        log.info("[checkpointer] usando InMemorySaver (REDIS_URL/POSTGRES_URL ausentes ou falharam)")
        self._shared = InMemorySaver()
        self._kind = "memory"
        return self._shared

    async def aclose(self) -> None:
        """Fecha pools no shutdown (FastAPI lifespan)."""
        if self._redis_ctx is not None and self._redis_obj is not None:
            try:
                await self._redis_ctx.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                log.warning("[checkpointer] erro ao fechar Redis: %s", exc)
        if self._postgres_ctx is not None and self._postgres_obj is not None:
            try:
                await self._postgres_ctx.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                log.warning("[checkpointer] erro ao fechar Postgres: %s", exc)
        self._shared = None
        self._postgres_ctx = None
        self._postgres_obj = None
        self._redis_ctx = None
        self._redis_obj = None


def _mask(url: str) -> str:
    """Esconde senha em URL."""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def thread_id_for(project_id: str, instance: str, phone: str) -> str:
    """Chave de checkpoint estável por lead."""
    return f"{project_id}:{instance}:{phone}"
