"""
Ferramentas auxiliares: parsing de tags, chunking de WhatsApp, cliente Evolution.

Equivalente ao `humanization.ts` + `evo-api.js` do projeto antigo, em Python.
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass

import httpx


# ────────────────────────────────────────────────────────────────────
# Parsing de tags secretas (mantém compat 1:1 com o bot antigo)
# ────────────────────────────────────────────────────────────────────

_RE_COMPROU = re.compile(r"\[\s*COMPROU\s*\]", re.IGNORECASE)
_RE_AGENDAR = re.compile(r"\[\s*AGENDAR\s*:\s*(\d+)\s*\]", re.IGNORECASE)
_RE_REACT   = re.compile(r"\[\s*REACT\s*:\s*([^\]\s]+)\s*\]", re.IGNORECASE)
_RE_QUOTE   = re.compile(r"\[\s*QUOTE\s*\]", re.IGNORECASE)


@dataclass
class ParsedTags:
    text: str
    has_converted: bool
    schedule_minutes: int | None
    react_emoji: str | None
    quote_previous: bool


def parse_tags(raw: str) -> ParsedTags:
    """Remove e captura todas as tags secretas da resposta da IA."""
    has_converted = bool(_RE_COMPROU.search(raw))

    schedule_minutes: int | None = None
    m = _RE_AGENDAR.search(raw)
    if m:
        # Mesmo clamp do bot antigo: min 5, max 10080 (7 dias).
        schedule_minutes = max(5, min(10080, int(m.group(1))))

    react_emoji: str | None = None
    m = _RE_REACT.search(raw)
    if m:
        react_emoji = m.group(1)

    quote_previous = bool(_RE_QUOTE.search(raw))

    text = raw
    text = _RE_COMPROU.sub("", text)
    text = _RE_AGENDAR.sub("", text)
    text = _RE_REACT.sub("", text)
    text = _RE_QUOTE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return ParsedTags(
        text=text,
        has_converted=has_converted,
        schedule_minutes=schedule_minutes,
        react_emoji=react_emoji,
        quote_previous=quote_previous,
    )


# ────────────────────────────────────────────────────────────────────
# Chunking em bolhas (replica humanization.chunkTextForWhatsApp)
# ────────────────────────────────────────────────────────────────────

def chunk_for_whatsapp(
    text: str,
    max_bubbles: int = 2,
    max_chars: int = 320,
) -> list[str]:
    """
    Quebra o texto em bolhas pequenas (regra: máx 2 bolhas, ~320 chars cada).

    1. Split em parágrafos (linhas em branco).
    2. Se um parágrafo passar de max_chars, split em sentenças.
    3. Cap em max_bubbles — sobra concatenado na última bolha.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    bubbles: list[str] = []

    for p in paragraphs:
        if len(p) <= max_chars:
            bubbles.append(p)
            continue
        # Split em sentenças preservando pontuação.
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sentences:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf = f"{buf} {s}"
            else:
                bubbles.append(buf)
                buf = s
        if buf:
            bubbles.append(buf)

    if len(bubbles) > max_bubbles:
        head = bubbles[: max_bubbles - 1]
        tail = " ".join(bubbles[max_bubbles - 1 :])
        bubbles = head + [tail]

    return bubbles


# ────────────────────────────────────────────────────────────────────
# Cliente Evolution API (mantém endpoints do bot antigo)
# ────────────────────────────────────────────────────────────────────

class EvolutionClient:
    """Cliente HTTP para a Evolution API (mesma API usada pelo bot antigo)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 20.0,
    ):
        self.base_url = (base_url or os.getenv("EVOLUTION_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("EVOLUTION_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"apikey": self.api_key, "Content-Type": "application/json"}

    async def _post(self, path: str, body: dict) -> dict:
        if not self.base_url:
            return {"success": False, "error": "EVOLUTION_API_URL não configurada"}
        return await self._post_with_retry(path, body)

    async def _post_with_retry(
        self,
        path: str,
        body: dict,
        max_attempts: int = 3,
        backoff_base: float = 0.5,
    ) -> dict:
        """
        POST com retry exponencial (0.5s, 1s, 2s) em falhas transitórias.
        Considera transitório: timeout, conexão recusada, status 5xx, 429.
        Não retenta 4xx (exceto 429), 401, 403.
        """
        import asyncio

        last_err: dict | None = None
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    r = await client.post(
                        f"{self.base_url}{path}", json=body, headers=self._headers()
                    )
                    try:
                        payload = r.json()
                    except Exception:
                        payload = {"raw": r.text}
                    if r.is_success:
                        return {"success": True, "status": r.status_code, **payload}
                    # 4xx (≠ 429) → não retenta
                    if r.status_code < 500 and r.status_code != 429:
                        return {"success": False, "status": r.status_code, **payload}
                    last_err = {"success": False, "status": r.status_code, **payload}
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                last_err = {"success": False, "error": str(exc), "type": exc.__class__.__name__}

            # Backoff exponencial antes da próxima tentativa.
            if attempt < max_attempts - 1:
                await asyncio.sleep(backoff_base * (2 ** attempt))

        return last_err or {"success": False, "error": "unknown"}

    async def send_text(self, instance: str, to: str, text: str) -> dict:
        return await self._post(f"/message/sendText/{instance}", {"number": to, "text": text})

    async def send_typing(self, instance: str, to: str, duration_ms: int | None = None) -> dict:
        body: dict = {"number": to, "presence": "composing"}
        if duration_ms:
            body["delay"] = duration_ms
        return await self._post(f"/chat/sendPresence/{instance}", body)

    async def send_reaction(self, instance: str, to: str, message_id: str, emoji: str) -> dict:
        body = {
            "reactionMessage": {
                "key": {"remoteJid": f"{to}@s.whatsapp.net", "id": message_id, "fromMe": False},
                "reaction": emoji,
            }
        }
        return await self._post(f"/message/sendReaction/{instance}", body)

    async def mark_read(self, instance: str, remote_jid: str, message_id: str) -> dict:
        body = {"readMessages": [{"remoteJid": remote_jid, "id": message_id, "fromMe": False}]}
        return await self._post(f"/chat/markMessageAsRead/{instance}", body)


# ────────────────────────────────────────────────────────────────────
# Simulação de digitação (replica simulateTyping com cps variável)
# ────────────────────────────────────────────────────────────────────

def typing_delay_ms(text: str, cps_base: float = 3.0) -> int:
    """Calcula quantos ms de typing antes de enviar o chunk."""
    chars = len(text)
    cps = max(2.0, min(9.0, cps_base * random.uniform(0.8, 1.2)))
    base = (chars / cps) * 1000
    return int(min(12000, max(1500, base * random.uniform(0.8, 1.2))))


def jitter_between_bubbles_ms() -> int:
    """Pausa randomizada entre bolhas (replica o setTimeout do bot antigo)."""
    return int(random.uniform(1500, 4000))
