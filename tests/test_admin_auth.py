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


# ────────────────────────────────────────────────────────────────────
# TenantResolver — bounded cache + no anon fallback
# ────────────────────────────────────────────────────────────────────

def test_tenant_resolver_cache_bounded(monkeypatch):
    """Cache TTL não cresce além de _CACHE_MAX."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-service-key")
    from memory.supabase_tenant import TenantResolver
    r = TenantResolver()
    # Override max pra testar rápido
    r._CACHE_MAX = 4
    import time
    for i in range(6):
        r._cache[f"inst_{i}"] = (f"proj_{i}", time.time() + 60)
        if len(r._cache) > r._CACHE_MAX:
            r._evict_oldest()
    # Após eviction, cache ≤ MAX
    assert len(r._cache) <= 4


def test_tenant_resolver_no_anon_fallback(monkeypatch):
    """Sem SERVICE_KEY e sem SUPABASE_ALLOW_ANON, key fica vazio."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ALLOW_ANON", raising=False)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    from memory.supabase_tenant import TenantResolver
    r = TenantResolver()
    assert r.key == ""
    assert r.enabled is False


def test_tenant_resolver_anon_opt_in(monkeypatch):
    """SUPABASE_ALLOW_ANON=1 permite fallback explícito."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_ALLOW_ANON", "1")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    from memory.supabase_tenant import TenantResolver
    r = TenantResolver()
    assert r.key == "anon-key"


# ────────────────────────────────────────────────────────────────────
# AuditLog — payload schema alinhado com migration 0003
# ────────────────────────────────────────────────────────────────────

async def test_audit_log_payload_schema(monkeypatch):
    """AuditLogRepo.write envia payload com campos corretos da migration 0003."""
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")
    monkeypatch.setattr("panel.repos.httpx.AsyncClient", FakeClient)

    from panel.repos import AuditLogRepo
    repo = AuditLogRepo()
    await repo.write(
        actor_id="user-1",
        actor_email="admin@example.com",
        action="section.patch",
        target_type="project_config",
        target_id="padrao",
        metadata={"section_key": "prices", "size": 1500},
    )
    # Campos devem alinhar com schema do 0003_flows_and_misc.sql
    payload = captured.get("json") or {}
    assert payload["admin_id"] == "user-1"
    assert payload["admin_email"] == "admin@example.com"
    assert payload["action"] == "section.patch"
    assert payload["resource_type"] == "project_config"
    assert payload["resource_id"] == "padrao"
    assert payload["after_state"] == {"section_key": "prices", "size": 1500}


async def test_audit_log_no_url_noop(monkeypatch):
    """Sem SUPABASE_URL, write é no-op (não levanta)."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    from panel.repos import AuditLogRepo
    repo = AuditLogRepo()
    # Não deve levantar
    await repo.write("u", "e@x.com", "a", "t", "i")


# ────────────────────────────────────────────────────────────────────
# CSRF middleware — exempt paths + token validation
# ────────────────────────────────────────────────────────────────────

def test_csrf_exempt_webhook_paths(monkeypatch):
    """Webhooks externos não passam por CSRF check."""
    monkeypatch.setenv("WEBHOOK_SECRET", "x" * 32)
    monkeypatch.setenv("EVOLUTION_API_KEY", "y" * 32)
    from main import _is_csrf_exempt
    assert _is_csrf_exempt("/webhook/evolution") is True
    assert _is_csrf_exempt("/webhook") is True
    assert _is_csrf_exempt("/api/trigger-followup") is True
    assert _is_csrf_exempt("/health") is True
    assert _is_csrf_exempt("/api/admin/projects") is False
    assert _is_csrf_exempt("/admin/agent") is False
