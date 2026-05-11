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
# Chunking humanizado em bolhas
# ────────────────────────────────────────────────────────────────────

# Padrões protegidos: NUNCA quebrar no meio destes (links, chaves PIX, etc).
# Ordem importa — combinações mais específicas primeiro.
_PROTECTED_PATTERNS = [
    r"https?://\S+",                                     # URLs http/https
    r"\bwww\.\S+",                                       # URLs bare www
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",                     # emails
    r"\b(?:\+?55\s?)?\(?\d{2,3}\)?\s?\d{4,5}-?\d{4}\b",  # telefones BR
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",  # UUID (pix)
    r"\b[A-Za-z0-9_-]{32,}\b",                           # chaves PIX random longas
    # R$ amounts incluindo ranges/listas — "R$40", "R$ 40,00", "R$40 a R$60", "R$40/mes"
    r"R\$\s*\d+(?:[.,]\d{1,3})*(?:\s*(?:a|e|–|-|/|por)\s*R\$?\s*\d+(?:[.,]\d{1,3})*)*",
    r"`[^`\n]+`",                                        # inline code
    r"```[\s\S]+?```",                                   # code blocks
]

_PROTECT_TOKEN = "\x00PROT{}\x00"
_PROTECT_RE = re.compile("|".join(f"(?:{p})" for p in _PROTECTED_PATTERNS))


def _protect_regions(text: str) -> tuple[str, list[str]]:
    """Substitui regiões protegidas por placeholders. Retorna (texto_seguro, lista_originais)."""
    matches: list[str] = []

    def _capture(m: re.Match) -> str:
        idx = len(matches)
        matches.append(m.group(0))
        return _PROTECT_TOKEN.format(idx)

    safe = _PROTECT_RE.sub(_capture, text)
    return safe, matches


def _restore_regions(text: str, matches: list[str]) -> str:
    """Substitui placeholders pelos originais."""
    for i, original in enumerate(matches):
        text = text.replace(_PROTECT_TOKEN.format(i), original)
    return text


# Conectivos brasileiros que marcam pausas naturais (split point opcional).
_CONNECTIVES = (
    "mas",
    "porém",
    "então",
    "olha",
    "tipo",
    "ou seja",
    "tipo assim",
    "vamos lá",
    "se liga",
    "agora",
    "aí",
)
_CONNECTIVE_RE = re.compile(
    r"\s+(?=(?:" + "|".join(re.escape(c) for c in _CONNECTIVES) + r")\b)",
    re.IGNORECASE,
)
_COMMA_BREAK_RE = re.compile(r"(?<=[,;:])\s+(?=[A-Za-zÀ-ÿ])")


def _split_smart(text: str, max_chars: int) -> list[str]:
    """
    Quebra texto longo em pedaços <= max_chars priorizando:
    1. Pontuação forte [.!?]
    2. Conectivos ("mas", "então", "porém"...)
    3. Vírgulas/dois-pontos
    4. Espaços (último recurso, palavra inteira)
    """
    if len(text) <= max_chars:
        return [text]

    # Sentenças
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)

    # Se alguma sentença sozinha excede max_chars, quebra por conectivos + vírgulas.
    refined: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            refined.append(c)
            continue
        pieces = _CONNECTIVE_RE.split(c)
        sub: list[str] = []
        for piece in pieces:
            if len(piece) <= max_chars:
                sub.append(piece)
            else:
                sub.extend(p for p in _COMMA_BREAK_RE.split(piece) if p)
        # Recombina sub-pedaços respeitando max_chars
        buf2 = ""
        for piece in sub:
            piece = piece.strip()
            if not piece:
                continue
            if not buf2:
                buf2 = piece
            elif len(buf2) + 1 + len(piece) <= max_chars:
                buf2 = f"{buf2} {piece}"
            else:
                refined.append(buf2)
                buf2 = piece
        if buf2:
            refined.append(buf2)
    return refined


def chunk_for_whatsapp(
    text: str,
    max_bubbles: int = 3,
    max_chars: int = 140,
    min_chars: int = 25,
) -> list[str]:
    """
    Quebra texto em bolhas humanizadas pro WhatsApp.

    Estratégia:
      1. Protege URLs, chaves PIX, telefones, R$ amounts, emails (nunca quebra dentro)
      2. Split em parágrafos (\\n\\n)
      3. Cada parágrafo grande → split por sentença → por conectivos → por vírgulas
      4. Junta bolhas minúsculas (<min_chars) com vizinha
      5. Limita a max_bubbles (sobra concatenada na última)

    Defaults: 3 bolhas × 140 chars — sensação humana, sem fragmentar demais.
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []

    # 1. Protege regiões intocáveis (URLs/PIX/R$/etc)
    safe, matches = _protect_regions(text)

    # 2. Split em parágrafos
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", safe) if p.strip()]

    # 3. Cada parágrafo → smart split
    bubbles: list[str] = []
    for p in paragraphs:
        bubbles.extend(_split_smart(p, max_chars))

    # 4. Merge bolhas curtas demais (< min_chars) com vizinha
    merged: list[str] = []
    for b in bubbles:
        b = b.strip()
        if not b:
            continue
        if merged and (len(b) < min_chars or len(merged[-1]) < min_chars):
            candidate = f"{merged[-1]} {b}"
            if len(candidate) <= int(max_chars * 1.25):
                merged[-1] = candidate
                continue
        merged.append(b)

    # 5. Cap em max_bubbles — sobra concatenada na última
    if len(merged) > max_bubbles:
        head = merged[: max_bubbles - 1]
        tail = " ".join(merged[max_bubbles - 1 :])
        merged = head + [tail]

    # 6. Restaura regiões protegidas
    return [_restore_regions(b, matches).strip() for b in merged if b.strip()]


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
