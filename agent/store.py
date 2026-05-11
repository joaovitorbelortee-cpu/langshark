"""
LangGraph Store (long-term memory cross-thread).

Diferença do checkpointer:
  - Checkpointer = state da thread/conversa (messages, intent, reply, etc).
  - Store        = memória persistente entre threads (lead_facts, preferências).

Padrão LangGraph oficial: nós acessam via `runtime.store.aput/aget/asearch`.
Namespaces: tuple de strings, ex: ("padrao", "5511999999999") por lead.

Hierarquia:
  1. REDIS_URL setado → AsyncRedisStore (com vector search opcional)
  2. Fallback → InMemoryStore (dev/teste — não persiste entre restarts)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.store.memory import InMemoryStore

log = logging.getLogger(__name__)


class StoreProvider:
    """
    Wrapper assíncrono que decide o store em runtime.

    Uso:
        provider = StoreProvider()
        store = await provider.shared()
        graph = build_graph(checkpointer=cp, store=store)
    """

    def __init__(self, redis_url: str | None = None):
        self.redis_url = (redis_url or os.getenv("REDIS_URL", "")).strip()
        self._shared: Any | None = None
        self._redis_ctx = None
        self._redis_obj: Any | None = None
        self._kind: str = "uninit"

    @property
    def kind(self) -> str:
        return self._kind

    async def shared(self) -> Any:
        """Singleton por processo."""
        if self._shared is not None:
            return self._shared

        if self.redis_url:
            try:
                from langgraph.store.redis.aio import AsyncRedisStore

                # TTL default 90 dias — lead facts não viram lixo eternamente,
                # mas duram bastante pra retomar conversas antigas
                ttl_days = int(os.getenv("STORE_TTL_DAYS", "90"))
                self._redis_ctx = AsyncRedisStore.from_conn_string(
                    self.redis_url,
                    ttl={"default_ttl": ttl_days * 24 * 60, "refresh_on_read": True},
                )
                self._redis_obj = await self._redis_ctx.__aenter__()
                try:
                    await self._redis_obj.asetup()
                except Exception as setup_exc:  # noqa: BLE001
                    log.warning("[store] asetup Redis falhou (idx talvez já existe): %s", setup_exc)
                self._shared = self._redis_obj
                self._kind = "redis"
                log.info("[store] AsyncRedisStore pronto (ttl=%dd)", ttl_days)
                return self._shared
            except ImportError as exc:
                log.warning("[store] langgraph-checkpoint-redis não instalado: %s", exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("[store] falha ao abrir Redis store: %s — fallback InMemory", exc)

        log.info("[store] usando InMemoryStore (REDIS_URL ausente ou falhou)")
        self._shared = InMemoryStore()
        self._kind = "memory"
        return self._shared

    async def aclose(self) -> None:
        if self._redis_ctx is not None and self._redis_obj is not None:
            try:
                await self._redis_ctx.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                log.warning("[store] erro ao fechar Redis: %s", exc)
        self._shared = None
        self._redis_ctx = None
        self._redis_obj = None


# ──────────────────────────────────────────────────────────────────────
# Helpers de namespace pra padronizar uso entre nós
# ──────────────────────────────────────────────────────────────────────

def lead_namespace(project_id: str, phone: str) -> tuple[str, ...]:
    """Namespace canônico pra dados de um lead específico.
    Ex: ("padrao", "5511999999999")"""
    return (project_id or "padrao", phone or "unknown")


def project_namespace(project_id: str) -> tuple[str, ...]:
    """Namespace pra dados cross-lead do projeto (ex: aprendizados globais)."""
    return (project_id or "padrao", "_project")


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton — set by main lifespan, lido pelos nós.
# Workaround simples vs depender de runtime.store injection do LangGraph
# (API muda entre versões 0.2 / 0.3).
# ──────────────────────────────────────────────────────────────────────

_shared_store: Any | None = None


def set_shared_store(store: Any | None) -> None:
    global _shared_store
    _shared_store = store


def get_shared_store() -> Any | None:
    return _shared_store
