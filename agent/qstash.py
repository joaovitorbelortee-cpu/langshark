"""
Upstash QStash scheduler para follow-ups.

Fluxo:
  1. Grafo termina e emite `schedule_minutes` via tag [AGENDAR: N] da IA
  2. main.py chama `schedule_followup()` → QStash agenda POST pro nosso webhook
  3. QStash dispara `POST /api/trigger-followup` após N minutos
  4. Endpoint roda o grafo com `intent="follow_up"` injetado

KillSwitch: se cliente responde antes do disparo, marcamos last_message_from=lead
no Redis e o trigger-followup checa antes de enviar.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)


class QStashClient:
    """
    Cliente mínimo pra Upstash QStash REST API.

    Endpoint: https://qstash.upstash.io/v2/publish/{target_url}
    Header `Upstash-Delay: 30m` agenda o disparo.
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        target_base_url: str | None = None,
        timeout: float = 10.0,
    ):
        self.token = (token or os.getenv("QSTASH_TOKEN", "")).strip()
        self.base_url = (base_url or os.getenv("QSTASH_URL", "https://qstash.upstash.io")).rstrip("/")
        self.target_base = (target_base_url or os.getenv("PUBLIC_BASE_URL", "")).rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.target_base)

    async def schedule_followup(
        self,
        project_id: str,
        instance: str,
        phone: str,
        delay_minutes: int,
        push_name: str = "",
    ) -> dict[str, Any]:
        """
        Agenda POST pro nosso `/api/trigger-followup`.

        Retorna dict com {ok, message_id?, error?}.
        Se QStash não estiver configurado, retorna {ok: False, skipped: True}.
        """
        if not self.enabled:
            return {"ok": False, "skipped": True, "reason": "qstash not configured"}

        # Clamp seguro: min 1min (lead diz "em 3 min"), max 7 dias.
        # QStash suporta Upstash-Delay em segundos também via formato "30s".
        delay_minutes = max(1, min(10080, int(delay_minutes)))

        # Quando dispara, QStash POST pro target. Pra autenticar, forward o
        # WEBHOOK_SECRET como header `apikey` (formato que _check_auth aceita).
        # QSTASH_TOKEN não serve aqui — é credencial DO QStash, não do nosso bot.
        webhook_secret = os.getenv("WEBHOOK_SECRET", "").strip()

        target = f"{self.target_base}/api/trigger-followup"
        payload = {
            "project_id": project_id,
            "instance_name": instance,
            "phone": phone,
            "push_name": push_name,
        }

        headers = {
            "Authorization": f"Bearer {self.token}",  # auth pra publicar no QStash
            "Content-Type": "application/json",
            "Upstash-Delay": f"{delay_minutes}m",
        }
        # Forward apikey pro trigger-followup endpoint (auth nossa)
        if webhook_secret:
            headers["Upstash-Forward-apikey"] = webhook_secret

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    f"{self.base_url}/v2/publish/{target}",
                    headers=headers,
                    json=payload,
                )
                r.raise_for_status()
                data = r.json() if r.content else {}
                return {"ok": True, "message_id": data.get("messageId"), "delay_min": delay_minutes}
        except (httpx.HTTPError, ValueError) as exc:
            # HTTPError: network/timeout/status; ValueError: JSON decode falhou
            log.warning("[qstash] schedule_followup falhou: %s", exc)
            return {"ok": False, "error": str(exc), "delay_min": delay_minutes}


def verify_qstash_signature(_signature: str | None, _body_bytes: bytes) -> bool:
    """
    Verificação HMAC do QStash. Em produção, valide assinaturas via QSTASH_CURRENT_SIGNING_KEY
    + QSTASH_NEXT_SIGNING_KEY (rotação). Implementação básica: confiamos no token compartilhado
    no header `Authorization`.

    Para validação completa de signature, plug no `qstash-python` SDK ou implemente
    JWT verification com as chaves rotativas.
    """
    # Placeholder — main.py valida via WEBHOOK_SECRET no header
    return True
