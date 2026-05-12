"""
Flow executor — dispara sequência cadastrada via cliente de envio (canal agnostic).

Independente de framework. Recebe um `MessageSender` protocol que tem:
  - send_text(to, text) → enviar texto
  - send_typing(to, duration_ms) → simular digitação (opcional)
  - send_media(to, kind, url, caption=None, file_name=None) → enviar mídia

Adapta pra qualquer canal (WhatsApp/Telegram/SMS/Discord/etc).
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Protocol

from .flows import Flow, get_flow

log = logging.getLogger(__name__)


class MessageSender(Protocol):
    """Interface mínima do cliente de envio."""

    async def send_text(self, to: str, text: str) -> Any: ...

    async def send_typing(self, to: str, duration_ms: int = 1200) -> Any:
        """Opcional. Se não suportado, pode ser no-op."""
        ...

    async def send_media(
        self,
        to: str,
        kind: str,         # "image" | "video" | "audio" | "document"
        url: str,
        caption: str | None = None,
        file_name: str | None = None,
    ) -> Any: ...


async def execute_flow(
    project_id: str,
    flow_name: str,
    to: str,
    sender: MessageSender,
    step_delay_range: tuple[float, float] = (1.5, 2.5),
    typing_duration_ms: int = 1200,
    typing_delay_s: float = 1.2,
) -> dict[str, Any]:
    """
    Dispara flow `flow_name` pra `to` via sender.

    Args:
        project_id: tenant key (multi-projeto)
        flow_name: nome do flow (case-insensitive)
        to: destinatário (phone, chat_id, etc — canal-specific)
        sender: cliente que implementa MessageSender Protocol
        step_delay_range: random delay entre steps (humanização)
        typing_duration_ms: duração typing indicator pra steps text
        typing_delay_s: sleep após typing antes do send (humanização)

    Returns:
        {
          "dispatched": bool,
          "sent_count": int,
          "flow_name": str,
          "errors": list[str]  # exceptions individuais por step
        }

    Erros por step não derrubam o flow inteiro (best-effort).
    Se NENHUM step foi enviado, dispatched=False (fallback pro caller decidir).
    """
    flow: Flow | None = get_flow(project_id, flow_name)
    if not flow:
        return {"dispatched": False, "sent_count": 0, "flow_name": flow_name, "errors": ["flow not found"]}

    sent = 0
    errors: list[str] = []

    for i, step in enumerate(flow.steps):
        if i > 0:
            jitter = random.uniform(*step_delay_range)
            await asyncio.sleep(jitter)

        kind = (step.get("type") or "").lower()
        try:
            if kind == "text":
                content = step.get("content", "")
                if not content:
                    continue
                try:
                    await sender.send_typing(to, duration_ms=typing_duration_ms)
                    await asyncio.sleep(typing_delay_s)
                except Exception:  # noqa: BLE001
                    pass  # typing opcional
                await sender.send_text(to, content)
                sent += 1
            elif kind in ("image", "video", "audio", "document"):
                url = step.get("url") or step.get("filePath")
                if not url:
                    errors.append(f"step {i} ({kind}): sem url")
                    continue
                await sender.send_media(
                    to=to,
                    kind=kind,
                    url=url,
                    caption=step.get("caption", "") or None,
                    file_name=step.get("fileName") if kind == "document" else None,
                )
                sent += 1
            else:
                errors.append(f"step {i}: tipo desconhecido '{kind}'")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("[flow] step %s falhou: %s", kind, exc)
            errors.append(f"step {i} ({kind}): {exc}")
            continue

    # Fallback safety — todos falharam = caller pode decidir reagir
    dispatched = sent > 0
    if not dispatched:
        log.warning("[flow] '%s' → %s: nenhum step enviado (%d erros)", flow.name, to, len(errors))

    return {
        "dispatched": dispatched,
        "sent_count": sent,
        "flow_name": flow.name,
        "errors": errors,
    }
