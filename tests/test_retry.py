"""Testa retry exponencial do EvolutionClient."""
from __future__ import annotations

import httpx
import pytest

from agent.tools import EvolutionClient


pytestmark = pytest.mark.asyncio


class _Resp:
    def __init__(self, status: int, json_data: dict | None = None):
        self.status_code = status
        self.is_success = 200 <= status < 300
        self._json = json_data or {}
        self.text = ""

    def json(self):
        return self._json


async def test_retry_succeeds_after_transient_5xx(monkeypatch):
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                return _Resp(503)
            return _Resp(200, {"id": "ok"})

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    evo = EvolutionClient(base_url="http://fake", api_key="k")
    r = await evo.send_text("inst", "5511", "hi")
    assert r["success"] is True
    assert calls["n"] == 3


async def test_retry_gives_up_after_max_attempts(monkeypatch):
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k):
            calls["n"] += 1
            return _Resp(500)

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    evo = EvolutionClient(base_url="http://fake", api_key="k")
    r = await evo.send_text("inst", "5511", "hi")
    assert r["success"] is False
    assert calls["n"] == 3


async def test_retry_skips_4xx(monkeypatch):
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k):
            calls["n"] += 1
            return _Resp(404)

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    evo = EvolutionClient(base_url="http://fake", api_key="k")
    r = await evo.send_text("inst", "5511", "hi")
    assert r["success"] is False
    assert r["status"] == 404
    assert calls["n"] == 1  # não retentou


async def test_retry_429_is_retried(monkeypatch):
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k):
            calls["n"] += 1
            if calls["n"] < 2:
                return _Resp(429)
            return _Resp(200, {})

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    evo = EvolutionClient(base_url="http://fake", api_key="k")
    r = await evo.send_text("inst", "5511", "hi")
    assert r["success"] is True
    assert calls["n"] == 2
