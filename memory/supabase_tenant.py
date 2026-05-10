"""
Adapter Supabase pra resolver project_id a partir de instance_name.

Replica `instance_projects` do schema antigo:
    SELECT project_id FROM instance_projects WHERE instance_name = $1 LIMIT 1

Funciona em fallback se SUPABASE_URL/KEY estiverem ausentes — devolve None e o
caller decide (usar DEFAULT_PROJECT_ID via env).

Cache local em memória com TTL pra evitar 1 query por mensagem.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


class TenantResolver:
    def __init__(
        self,
        url: str | None = None,
        service_key: str | None = None,
        cache_ttl_seconds: int = 300,
        timeout: float = 5.0,
    ):
        self.url = (url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.key = service_key or os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
        self.timeout = timeout
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[str | None, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    async def resolve(self, instance_name: str) -> str | None:
        """Devolve project_id ou None."""
        if not instance_name:
            return None
        # Cache hit
        cached = self._cache.get(instance_name)
        if cached and cached[1] > time.time():
            return cached[0]

        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(
                    f"{self.url}/rest/v1/instance_projects",
                    params={
                        "select": "project_id",
                        "instance_name": f"eq.{instance_name}",
                        "limit": "1",
                    },
                    headers={
                        "apikey": self.key,
                        "Authorization": f"Bearer {self.key}",
                        "Accept": "application/json",
                    },
                )
                r.raise_for_status()
                data: list[dict[str, Any]] = r.json() or []
                project_id = data[0]["project_id"] if data else None
        except Exception as exc:  # noqa: BLE001
            log.warning("[tenant] falha ao consultar Supabase: %s", exc)
            project_id = None

        self._cache[instance_name] = (project_id, time.time() + self.cache_ttl)
        return project_id
