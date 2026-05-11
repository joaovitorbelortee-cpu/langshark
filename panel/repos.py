"""
Repositórios REST Supabase pro painel admin.

Sem ORM. Usa httpx contra Supabase REST API com SUPABASE_SERVICE_KEY (bypass RLS).
"""
from __future__ import annotations

import os
from typing import Any

import httpx


def _supabase_creds() -> tuple[str, str]:
    """
    Sempre exige SERVICE_KEY pro painel admin — anon_key não tem privilégio em RLS.
    Fallback pra anon só permitido se SUPABASE_ALLOW_ANON=1 (dev local).
    """
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
    if not key and os.getenv("SUPABASE_ALLOW_ANON") == "1":
        key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    return url, key


def _headers() -> dict[str, str]:
    _url, key = _supabase_creds()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# ────────────────────────────────────────────────────────────────────
# ProjectConfigRepo
# ────────────────────────────────────────────────────────────────────

class ProjectConfigRepo:
    """CRUD em public.project_config."""

    async def list(self) -> list[dict[str, Any]]:
        url, _ = _supabase_creds()
        if not url:
            return []
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{url}/rest/v1/project_config",
                params={"select": "*"},
                headers=_headers(),
            )
            r.raise_for_status()
            return r.json() or []

    async def fetch(self, project_id: str) -> dict[str, Any] | None:
        url, _ = _supabase_creds()
        if not url:
            return None
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{url}/rest/v1/project_config",
                params={"select": "*", "project_id": f"eq.{project_id}", "limit": "1"},
                headers=_headers(),
            )
            r.raise_for_status()
            data = r.json() or []
            return data[0] if data else None

    async def patch(self, project_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        url, _ = _supabase_creds()
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.patch(
                f"{url}/rest/v1/project_config",
                params={"project_id": f"eq.{project_id}"},
                headers={**_headers(), "Prefer": "return=representation"},
                json=fields,
            )
            r.raise_for_status()
            data = r.json() or []
            return data[0] if data else {}

    async def patch_section(self, project_id: str, section_key: str, content: str) -> dict[str, Any]:
        """Atualiza brain_sections->{section_key}->content via SQL JSONB merge."""
        cfg = await self.fetch(project_id) or {}
        sections = cfg.get("brain_sections") or {}
        if section_key not in sections:
            raise ValueError(f"Section '{section_key}' inexistente em brain_sections")
        sections[section_key] = {**sections[section_key], "content": content}
        return await self.patch(project_id, {"brain_sections": sections})


# ────────────────────────────────────────────────────────────────────
# AdminUsersRepo
# ────────────────────────────────────────────────────────────────────

class AdminUsersRepo:
    """admin_users — login + sessões."""

    async def by_email(self, email: str) -> dict[str, Any] | None:
        url, _ = _supabase_creds()
        if not url:
            return None
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{url}/rest/v1/admin_users",
                params={"select": "*", "email": f"eq.{email}", "limit": "1"},
                headers=_headers(),
            )
            r.raise_for_status()
            data = r.json() or []
            return data[0] if data else None

    async def by_id(self, user_id: str) -> dict[str, Any] | None:
        url, _ = _supabase_creds()
        if not url:
            return None
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{url}/rest/v1/admin_users",
                params={"select": "*", "id": f"eq.{user_id}", "limit": "1"},
                headers=_headers(),
            )
            r.raise_for_status()
            data = r.json() or []
            return data[0] if data else None

    async def create(
        self,
        email: str,
        password_hash: str,
        display_name: str = "",
        project_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        url, _ = _supabase_creds()
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(
                f"{url}/rest/v1/admin_users",
                headers={**_headers(), "Prefer": "return=representation"},
                json={
                    "email": email,
                    "password_hash": password_hash,
                    "display_name": display_name,
                    "project_ids": project_ids or [],
                },
            )
            r.raise_for_status()
            data = r.json() or []
            return data[0] if data else {}

    async def touch_login(self, user_id: str) -> None:
        from datetime import datetime, timezone
        url, _ = _supabase_creds()
        if not url:
            return
        async with httpx.AsyncClient(timeout=5.0) as c:
            try:
                r = await c.patch(
                    f"{url}/rest/v1/admin_users",
                    params={"id": f"eq.{user_id}"},
                    headers=_headers(),
                    json={"last_login_at": datetime.now(timezone.utc).isoformat()},
                )
                r.raise_for_status()
            except httpx.HTTPError:
                # Login bem-sucedido não deve falhar por touch_login flaky.
                pass


# ────────────────────────────────────────────────────────────────────
# AIModelsCatalogRepo
# ────────────────────────────────────────────────────────────────────

class AIModelsCatalogRepo:
    """Catálogo curado de modelos pra dropdown."""

    async def list(self, only_active: bool = True) -> list[dict[str, Any]]:
        url, _ = _supabase_creds()
        if not url:
            return []
        params = {"select": "*", "order": "sort_order.asc"}
        if only_active:
            params["active"] = "eq.true"
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{url}/rest/v1/ai_models_catalog",
                params=params,
                headers=_headers(),
            )
            r.raise_for_status()
            return r.json() or []


# ────────────────────────────────────────────────────────────────────
# Generic CRUD helper (flows, products)
# ────────────────────────────────────────────────────────────────────

class _SupaTable:
    """Wrapper genérico pra CRUD em qualquer tabela com PK 'id'."""

    def __init__(self, table: str):
        self.table = table

    async def list(self, project_id: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        url, _ = _supabase_creds()
        if not url:
            return []
        params: dict[str, str] = {"select": "*", "order": "created_at.desc"}
        if project_id:
            params["project_id"] = f"eq.{project_id}"
        for k, v in filters.items():
            params[k] = f"eq.{v}"
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{url}/rest/v1/{self.table}", params=params, headers=_headers())
            r.raise_for_status()
            return r.json() or []

    async def get(self, row_id: str) -> dict[str, Any] | None:
        url, _ = _supabase_creds()
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"{url}/rest/v1/{self.table}",
                params={"select": "*", "id": f"eq.{row_id}", "limit": "1"},
                headers=_headers(),
            )
            r.raise_for_status()
            d = r.json() or []
            return d[0] if d else None

    async def insert(self, row: dict[str, Any]) -> dict[str, Any]:
        url, _ = _supabase_creds()
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(
                f"{url}/rest/v1/{self.table}",
                headers={**_headers(), "Prefer": "return=representation"},
                json=row,
            )
            r.raise_for_status()
            d = r.json() or []
            return d[0] if d else {}

    async def patch(self, row_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        url, _ = _supabase_creds()
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.patch(
                f"{url}/rest/v1/{self.table}",
                params={"id": f"eq.{row_id}"},
                headers={**_headers(), "Prefer": "return=representation"},
                json=fields,
            )
            r.raise_for_status()
            d = r.json() or []
            return d[0] if d else {}

    async def delete(self, row_id: str) -> bool:
        url, _ = _supabase_creds()
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.delete(
                f"{url}/rest/v1/{self.table}",
                params={"id": f"eq.{row_id}"},
                headers=_headers(),
            )
            return r.is_success


class FlowsRepo(_SupaTable):
    def __init__(self): super().__init__("flows")


class ProductsRepo(_SupaTable):
    def __init__(self): super().__init__("products")
    # products tem PK text; override get/delete pra usar string id
    async def insert(self, row: dict[str, Any]) -> dict[str, Any]:
        # products PK não é uuid auto — força id
        if "id" not in row or not row["id"]:
            import uuid
            row["id"] = str(uuid.uuid4())
        return await super().insert(row)
