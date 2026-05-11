"""
Smoke tests pro multi-tenant RBAC + auth do painel admin.

Foco em unit-test puro (sem rede): helpers, cookie flags, NX dedup expiry.
Cobertura E2E REST fica pra integration suite.
"""
from __future__ import annotations

import time

import pytest


# ────────────────────────────────────────────────────────────────────
# _user_can_access_project — RBAC helper
# ────────────────────────────────────────────────────────────────────

def test_user_can_access_project_admin_global():
    """Sem project_ids = admin global, acessa tudo."""
    from panel.api import _user_can_access_project
    user = {"id": "u1", "project_ids": []}
    assert _user_can_access_project(user, "any-project") is True
    assert _user_can_access_project(user, "padrao") is True


def test_user_can_access_project_scoped():
    """User com lista acessa só seus projetos."""
    from panel.api import _user_can_access_project
    user = {"id": "u1", "project_ids": ["alpha", "beta"]}
    assert _user_can_access_project(user, "alpha") is True
    assert _user_can_access_project(user, "beta") is True
    assert _user_can_access_project(user, "gamma") is False
    assert _user_can_access_project(user, "") is False


def test_user_can_access_project_missing_key():
    """Missing project_ids key tratado como admin global (defensivo)."""
    from panel.api import _user_can_access_project
    user = {"id": "u1"}  # sem project_ids
    assert _user_can_access_project(user, "any") is True


# ────────────────────────────────────────────────────────────────────
# _pick_current_project — view-level project resolution
# ────────────────────────────────────────────────────────────────────

def test_pick_current_project_respects_user_scope():
    """Query string fora do escopo do user não é honrada."""
    from panel.views import _pick_current_project
    projects = [
        {"project_id": "alpha", "agent_name": "A"},
        {"project_id": "beta",  "agent_name": "B"},
        {"project_id": "gamma", "agent_name": "C"},
    ]
    user = {"project_ids": ["alpha", "beta"]}
    # Tenta pegar gamma mas não está no escopo → cai pro default (alpha)
    pick = _pick_current_project(projects, requested="gamma", user=user)
    assert pick is not None
    assert pick["project_id"] in {"alpha", "beta"}


def test_pick_current_project_no_user():
    """Sem user = admin global, escolhe padrao se existe."""
    from panel.views import _pick_current_project
    projects = [
        {"project_id": "x", "agent_name": "X"},
        {"project_id": "padrao", "agent_name": "P"},
    ]
    pick = _pick_current_project(projects, requested=None, user=None)
    assert pick["project_id"] == "padrao"


def test_pick_current_project_empty_pool():
    """User sem nenhum projeto retorna None."""
    from panel.views import _pick_current_project
    projects = [{"project_id": "alpha"}, {"project_id": "beta"}]
    user = {"project_ids": ["zzz"]}
    assert _pick_current_project(projects, None, user) is None


# ────────────────────────────────────────────────────────────────────
# RedisStore._local_cmd — NX expiry semantics
# ────────────────────────────────────────────────────────────────────

def test_local_nx_respects_ttl_expiry():
    """SET NX sucede após TTL expirar (anti-dedup-falso-positivo permanente)."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")  # força local fallback
    # 1ª set com TTL curto
    r1 = store._local_cmd(("SET", "k1", "v1", "EX", "0.05", "NX"))
    assert r1 == "OK"
    # 2ª set NX imediato → deve falhar (chave viva)
    r2 = store._local_cmd(("SET", "k1", "v2", "EX", "10", "NX"))
    assert r2 is None
    # Espera expirar
    time.sleep(0.1)
    # 3ª set NX após expirar → deve suceder
    r3 = store._local_cmd(("SET", "k1", "v3", "EX", "10", "NX"))
    assert r3 == "OK"
    # Confirma valor sobrescrito
    assert store._local_cmd(("GET", "k1")) == "v3"


def test_local_nx_without_ttl():
    """NX sem TTL persiste indefinidamente."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    r1 = store._local_cmd(("SET", "perm", "v1", "NX"))
    assert r1 == "OK"
    r2 = store._local_cmd(("SET", "perm", "v2", "NX"))
    assert r2 is None  # chave permanente bloqueia NX


# ────────────────────────────────────────────────────────────────────
# Auth cookie flags — assert strict SameSite
# ────────────────────────────────────────────────────────────────────

def test_session_cookie_flags(monkeypatch):
    """Cookie session deve ter Secure + HttpOnly + SameSite=strict."""
    monkeypatch.setenv("WEBHOOK_SECRET", "x" * 32)
    from fastapi import Response

    from panel.auth import COOKIE_NAME, set_session_cookie

    resp = Response()
    set_session_cookie(resp, "fake.token.value")
    cookie_header = resp.raw_headers
    # Localiza set-cookie
    raw = [v for k, v in cookie_header if k.lower() == b"set-cookie"]
    assert raw, "set-cookie ausente"
    val = raw[0].decode()
    assert COOKIE_NAME in val
    assert "HttpOnly" in val
    assert "Secure" in val
    assert "samesite=strict" in val.lower()


# ────────────────────────────────────────────────────────────────────
# Cache eviction — empty dict guard
# ────────────────────────────────────────────────────────────────────

def test_cache_evict_empty_no_crash():
    """_evict_oldest em dict vazio não levanta ValueError."""
    from panel.cache import ProjectConfigCache
    c = ProjectConfigCache()
    # Não deve crashar mesmo sem entradas
    c._evict_oldest()
    assert c._store == {}
