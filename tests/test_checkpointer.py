"""
Testa o CheckpointerProvider.

- InMemorySaver fallback: sempre exercitado (sem deps).
- AsyncPostgresSaver: skip se POSTGRES_URL ausente. Roda em CI/Railway com URL real.
"""
from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.checkpointer import CheckpointerProvider, thread_id_for


async def test_inmemory_fallback_when_no_postgres():
    """Sem POSTGRES_URL → InMemorySaver."""
    provider = CheckpointerProvider(postgres_url="")
    cp = await provider.shared()
    assert isinstance(cp, InMemorySaver)
    assert provider.kind == "memory"
    await provider.aclose()


async def test_inmemory_fallback_when_postgres_unreachable(monkeypatch):
    """POSTGRES_URL inválida → fallback silencioso pra InMemorySaver."""
    provider = CheckpointerProvider(postgres_url="postgresql://invalid:invalid@127.0.0.1:1/none")
    cp = await provider.shared()
    # Mesmo configurado, conexão falha → cai pra in-memory.
    assert isinstance(cp, InMemorySaver)
    await provider.aclose()


def test_thread_id_format():
    """Chave de checkpoint estável."""
    tid = thread_id_for("padrao", "botzap", "5511999999999")
    assert tid == "padrao:botzap:5511999999999"


@pytest.mark.skipif(
    not os.getenv("POSTGRES_URL"),
    reason="POSTGRES_URL not set — skip live Postgres integration test",
)
async def test_postgres_checkpointer_live():
    """Integration: roda só se POSTGRES_URL estiver setada (Railway, dev local com docker)."""
    provider = CheckpointerProvider(postgres_url=os.getenv("POSTGRES_URL"))
    cp = await provider.shared()
    assert provider.kind == "postgres"
    # Smoke: setup já rodou, checkpointer está pronto.
    assert cp is not None
    await provider.aclose()
