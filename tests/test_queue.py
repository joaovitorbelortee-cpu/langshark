"""
Testes da FIFO queue Redis-backed.

Crítico: bot processa 1 msg por vez globalmente — anti-ban WhatsApp.
Paralelismo dispara N envios simultâneos do mesmo número → suspensão.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def store():
    """RedisStore com fallback in-memory (url='' força local)."""
    from memory.redis_store import RedisStore
    return RedisStore(url="", token="")


# ────────────────────────────────────────────────────────────────────
# Enqueue/dequeue FIFO ordering
# ────────────────────────────────────────────────────────────────────

async def test_queue_fifo_order(store):
    """Primeira mensagem enqueued é primeira dequeued."""
    await store.enqueue_message({"id": 1, "msg": "primeira"})
    await store.enqueue_message({"id": 2, "msg": "segunda"})
    await store.enqueue_message({"id": 3, "msg": "terceira"})

    m1 = await store.dequeue_message()
    m2 = await store.dequeue_message()
    m3 = await store.dequeue_message()

    assert m1["id"] == 1
    assert m2["id"] == 2
    assert m3["id"] == 3


async def test_queue_empty_returns_none(store):
    """RPOP em queue vazia retorna None."""
    assert await store.dequeue_message() is None


async def test_queue_length_tracks(store):
    """LLEN reflete tamanho real."""
    assert await store.queue_length() == 0
    await store.enqueue_message({"id": 1})
    await store.enqueue_message({"id": 2})
    assert await store.queue_length() == 2
    await store.dequeue_message()
    assert await store.queue_length() == 1
    await store.dequeue_message()
    assert await store.queue_length() == 0


async def test_queue_enqueue_returns_size(store):
    """enqueue_message retorna tamanho pós-LPUSH."""
    s1 = await store.enqueue_message({"id": 1})
    s2 = await store.enqueue_message({"id": 2})
    s3 = await store.enqueue_message({"id": 3})
    assert s1 == 1
    assert s2 == 2
    assert s3 == 3


# ────────────────────────────────────────────────────────────────────
# Requeue head — coloca de volta pra ser próximo a sair
# ────────────────────────────────────────────────────────────────────

async def test_requeue_head_pops_next(store):
    """Item requeued sai ANTES dos enqueued normais."""
    await store.enqueue_message({"id": "a"})
    await store.enqueue_message({"id": "b"})
    await store.requeue_head({"id": "priority"})
    # priority deve sair primeiro (foi pro fim RPUSH, RPOP pega de lá)
    first = await store.dequeue_message()
    assert first["id"] == "priority"
    # a, b mantêm ordem FIFO original
    assert (await store.dequeue_message())["id"] == "a"
    assert (await store.dequeue_message())["id"] == "b"


# ────────────────────────────────────────────────────────────────────
# Concurrency safety — múltiplos enqueue paralelo mantém todas
# ────────────────────────────────────────────────────────────────────

async def test_concurrent_enqueues_dont_lose_messages(store):
    """10 enqueues paralelos resultam em 10 itens na queue."""
    async def enq(i: int) -> None:
        await store.enqueue_message({"id": i})

    await asyncio.gather(*[enq(i) for i in range(10)])
    assert await store.queue_length() == 10

    seen = set()
    for _ in range(10):
        msg = await store.dequeue_message()
        seen.add(msg["id"])
    assert seen == set(range(10))


# ────────────────────────────────────────────────────────────────────
# Payload integrity
# ────────────────────────────────────────────────────────────────────

async def test_payload_roundtrip_preserves_nested_structure(store):
    """Payload com dict aninhado sobrevive serialização JSON."""
    payload = {
        "kind": "inbound",
        "instance": "botzap",
        "phone": "5511999999999",
        "message_id": "msg_abc123",
        "enqueued_at": 1700000000.5,
        "initial_state": {
            "project_id": "padrao",
            "user_message": "Olá, quanto custa?",
            "media_mime": None,
            "messages": [],
        },
    }
    await store.enqueue_message(payload)
    out = await store.dequeue_message()
    assert out == payload
    assert out["initial_state"]["user_message"] == "Olá, quanto custa?"


async def test_dequeue_malformed_returns_none():
    """Se JSON corrompido na queue, dequeue retorna None (não levanta)."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    # Empurra raw inválido direto no fallback
    store._fallback[store._QUEUE_KEY] = (["{ corrupted json"], None)
    out = await store.dequeue_message()
    assert out is None
