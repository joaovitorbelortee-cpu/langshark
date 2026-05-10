"""
Fixtures pytest: mocka LLM, EvolutionClient, Redis, RAG, Tenant.

Roda os testes em ambiente totalmente isolado (sem rede, sem Postgres).
"""
from __future__ import annotations

import warnings

# Suprime warning at-import da langgraph antes de qualquer import dessa lib.
warnings.filterwarnings("ignore", message="The default value of .allowed_objects.")

import asyncio  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from agent import nodes as nodes_mod  # noqa: E402
from agent.tools import EvolutionClient  # noqa: E402
from memory.redis_store import RedisStore  # noqa: E402
from memory.supabase_tenant import TenantResolver  # noqa: E402
from rag.catalog import CatalogRAG  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# Fakes
# ────────────────────────────────────────────────────────────────────

class FakeAIMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Substitui ChatOpenAI. Cada chamada devolve a próxima resposta scriptada."""

    def __init__(self, scripted: list[str] | None = None):
        self.scripted = list(scripted or [])
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> FakeAIMessage:
        self.calls.append(messages)
        if not self.scripted:
            return FakeAIMessage("ok [AGENDAR: 30]")
        return FakeAIMessage(self.scripted.pop(0))


class FakeEvolution(EvolutionClient):
    """Captura chamadas em vez de fazer HTTP."""

    def __init__(self):
        super().__init__(base_url="http://fake", api_key="fake")
        self.sent_text: list[tuple[str, str, str]] = []
        self.sent_typing: list[tuple[str, str, int]] = []
        self.sent_reaction: list[tuple[str, str, str, str]] = []

    async def send_text(self, instance, to, text):
        self.sent_text.append((instance, to, text))
        return {"success": True, "status": 200}

    async def send_typing(self, instance, to, duration_ms=None):
        self.sent_typing.append((instance, to, duration_ms or 0))
        return {"success": True}

    async def send_reaction(self, instance, to, message_id, emoji):
        self.sent_reaction.append((instance, to, message_id, emoji))
        return {"success": True}

    async def mark_read(self, instance, remote_jid, message_id):
        return {"success": True}


class FakeRedis(RedisStore):
    """Override que ignora a rede e usa só o fallback in-memory."""

    def __init__(self):
        # bypass init normal — força modo local
        self.url = ""
        self.token = ""
        self.timeout = 1.0
        self._fallback: dict[str, tuple[Any, float | None]] = {}


class FakeRAG(CatalogRAG):
    """ChromaDB stub: devolve hits fixos. Sem disco."""

    def __init__(self, hits: list[dict] | None = None):
        self.hits = hits or []

    def search(self, project_id, query, top_k=4):  # type: ignore[override]
        return list(self.hits)

    def format_context(self, hits):  # type: ignore[override]
        if not hits:
            return ""
        return "<catalogo_test>" + ", ".join(h["name"] for h in hits) + "</catalogo_test>"


class FakeTenant(TenantResolver):
    def __init__(self, mapping: dict[str, str] | None = None):
        self.url = ""
        self.key = ""
        self.cache_ttl = 0
        self.timeout = 1.0
        self._cache = {}
        self.mapping = mapping or {}

    @property
    def enabled(self) -> bool:  # type: ignore[override]
        return True

    async def resolve(self, instance_name: str) -> str | None:  # type: ignore[override]
        return self.mapping.get(instance_name)


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_llm():
    """Mocka _make_llm em agent.nodes pra devolver um FakeLLM scriptado."""
    holder: dict[str, FakeLLM] = {}

    def install(scripted: list[str] | None = None) -> FakeLLM:
        llm = FakeLLM(scripted)
        holder["llm"] = llm
        nodes_mod._make_llm = lambda **_kw: llm  # type: ignore[assignment]
        return llm

    install([])
    return install


@pytest.fixture
def fake_evolution():
    evo = FakeEvolution()
    nodes_mod.set_evolution(evo)
    return evo


@pytest.fixture
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(nodes_mod, "_redis", r)
    monkeypatch.setattr(nodes_mod, "get_redis", lambda: r)
    return r


@pytest.fixture
def fake_rag(monkeypatch):
    rag = FakeRAG([{"id": "p1", "name": "Plano X", "description": "desc", "price": 99.0, "score": 0.9}])
    monkeypatch.setattr(nodes_mod, "_rag", rag)
    monkeypatch.setattr(nodes_mod, "get_rag", lambda: rag)
    return rag


@pytest.fixture
def fake_tenant(monkeypatch):
    t = FakeTenant({"botzap": "padrao"})
    monkeypatch.setattr(nodes_mod, "_tenant", t)
    monkeypatch.setattr(nodes_mod, "get_tenant_resolver", lambda: t)
    return t


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
