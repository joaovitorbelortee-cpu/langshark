"""
Upstash QStash scheduler para follow-ups.

Fluxo:
  1. Bot decide schedule_minutes (via strategist.classify_lead)
  2. App chama schedule_followup() → QStash agenda POST pro webhook
  3. QStash dispara POST {target_base}/api/trigger-followup após N min
  4. Endpoint roda lógica de follow-up

KillSwitch: lead responder antes → app marca last_message_from=lead em storage
(Redis recomendado). Trigger checa antes de enviar.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)


class QStashClient:
    """
    Cliente mínimo Upstash QStash REST API.

    Endpoint: https://qstash.upstash.io/v2/publish/{target_url}
    Header `Upstash-Delay: 30m` agenda disparo.
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        target_base_url: str | None = None,
        webhook_secret: str | None = None,
        timeout: float = 10.0,
    ):
        self.token = (token or os.getenv("QSTASH_TOKEN", "")).strip()
        self.base_url = (base_url or os.getenv("QSTASH_URL", "https://qstash.upstash.io")).rstrip("/")
        self.target_base = (target_base_url or os.getenv("PUBLIC_BASE_URL", "")).rstrip("/")
        # Webhook secret pra forwarding como `apikey` header no callback
        self.webhook_secret = (webhook_secret or os.getenv("WEBHOOK_SECRET", "")).strip()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.target_base)

    async def schedule_followup(
        self,
        delay_minutes: int,
        payload: dict[str, Any],
        target_path: str = "/api/trigger-followup",
    ) -> dict[str, Any]:
        """
        Agenda POST genérico pro webhook do app.

        Args:
            delay_minutes: clamp [1, 10080] (1min - 7 dias)
            payload: dict JSON enviado pro endpoint quando dispara
            target_path: rota do endpoint do app (default /api/trigger-followup)

        Returns:
            {ok: bool, message_id?: str, error?: str, delay_min: int}
            Se QStash não configurado, {ok: False, skipped: True}
        """
        if not self.enabled:
            return {"ok": False, "skipped": True, "reason": "qstash not configured"}

        delay_minutes = max(1, min(10080, int(delay_minutes)))
        target = f"{self.target_base}{target_path}"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Upstash-Delay": f"{delay_minutes}m",
        }
        if self.webhook_secret:
            headers["Upstash-Forward-apikey"] = self.webhook_secret

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    f"{self.base_url}/v2/publish/{target}",
                    headers=headers,
                    json=payload,
                )
                r.raise_for_status()
                data = r.json() if r.content else {}
                return {
                    "ok": True,
                    "message_id": data.get("messageId"),
                    "delay_min": delay_minutes,
                }
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("[qstash] schedule falhou: %s", exc)
            return {"ok": False, "error": str(exc), "delay_min": delay_minutes}
