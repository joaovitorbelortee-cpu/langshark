"""
Ferramentas auxiliares: parsing de tags, chunking de WhatsApp, cliente Evolution.

Equivalente ao `humanization.ts` + `evo-api.js` do projeto antigo, em Python.
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any

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
    max_bubbles: int = 4,
    max_chars: int = 110,
    min_chars: int = 25,
    force_sentence_split: bool = True,
) -> list[str]:
    """
    Quebra texto em bolhas humanizadas pro WhatsApp.

    Estratégia:
      1. Protege URLs, chaves PIX, telefones, R$ amounts, emails (nunca quebra dentro)
      2. Split em parágrafos (\\n\\n)
      3. Force sentence split — se há 2+ frases E texto > 50 chars, split por frase
         mesmo se cabe em max_chars (mais humano).
      4. Cada parágrafo grande → split por sentença → conectivos → vírgulas
      5. Junta bolhas minúsculas (<min_chars) com vizinha
      6. Limita a max_bubbles (sobra concatenada na última)

    Defaults agressivos: 4 bolhas × 110 chars — fragmenta mais pra parecer humano.
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

    # 3. Cada parágrafo → smart split (com force_sentence_split)
    bubbles: list[str] = []
    for p in paragraphs:
        if force_sentence_split and len(p) > 50:
            # Força split por sentença mesmo se cabe em max_chars
            sentences = re.split(r"(?<=[.!?])\s+", p)
            if len(sentences) >= 2:
                # Cada sentença vira candidata a bolha (será merge se < min_chars)
                for s in sentences:
                    s = s.strip()
                    if s:
                        # Se sentença muito grande, ainda split por conectivos
                        if len(s) > max_chars:
                            bubbles.extend(_split_smart(s, max_chars))
                        else:
                            bubbles.append(s)
                continue
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

    async def send_text(
        self,
        instance: str,
        to: str,
        text: str,
        quoted_msg_id: str | None = None,
        quoted_msg_text: str | None = None,
    ) -> dict:
        """
        Envia texto. Se quoted_msg_id fornecido, usa o "Responder" do WhatsApp
        (formato Evolution v2 `quoted` field).
        """
        body: dict[str, Any] = {"number": to, "text": text}
        if quoted_msg_id:
            quoted_block: dict[str, Any] = {
                "key": {
                    "remoteJid": f"{to}@s.whatsapp.net",
                    "fromMe": False,
                    "id": quoted_msg_id,
                },
            }
            if quoted_msg_text:
                quoted_block["message"] = {"conversation": quoted_msg_text[:500]}
            body["quoted"] = quoted_block
        return await self._post(f"/message/sendText/{instance}", body)

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

    async def mark_messages_read(
        self,
        instance: str,
        remote_jid: str,
        message_ids: list[str],
    ) -> dict:
        """Marca MÚLTIPLAS msgs como lidas numa única call (anti rajada unread)."""
        if not message_ids:
            return {"success": True, "skipped": "no_ids"}
        body = {
            "readMessages": [
                {"remoteJid": remote_jid, "id": mid, "fromMe": False}
                for mid in message_ids
                if mid
            ],
        }
        if not body["readMessages"]:
            return {"success": True, "skipped": "empty"}
        return await self._post(f"/chat/markMessageAsRead/{instance}", body)

    async def set_settings(self, instance: str, **settings: Any) -> dict:
        """
        Atualiza settings da instância via /settings/set/{instance}.

        Settings relevantes pra blue ticks:
          - readMessages: True   → bot envia read receipts (✓✓ azul)
          - alwaysOnline: True   → instance fica online → delivery ✓✓ rápido
          - readStatus: True     → lê status (stories) — opcional
          - rejectCall: True     → rejeita chamadas automaticamente
          - msgCall: str         → msg quando rejeita chamada
          - groupsIgnore: True   → ignora msgs de grupo
          - syncFullHistory: False → não sincroniza histórico completo

        Body Evolution v2: dict plano com os campos acima.
        """
        return await self._post(f"/settings/set/{instance}", dict(settings))

    async def get_settings(self, instance: str) -> dict:
        """GET /settings/find/{instance} — retorna config atual."""
        if not self.base_url:
            return {"success": False, "error": "EVOLUTION_API_URL não configurada"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(
                f"{self.base_url}/settings/find/{instance}",
                headers=self._headers(),
            )
            try:
                return r.json() if r.is_success else {"success": False, "status": r.status_code}
            except Exception:  # noqa: BLE001
                return {"success": False, "raw": r.text[:200]}


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
