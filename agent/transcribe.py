"""
Whisper audio transcription via OpenAI API.

Convert audioMessage do Evolution (base64) em texto. Usado por vision_node
quando lead manda áudio — bot consome texto transcrito como user_message
normal e responde com contexto real.

Endpoint: POST /v1/audio/transcriptions (OpenAI direto).
OpenRouter pode proxy via mesmo path se key suportar — fallback automático.

Fail-soft: se Whisper falhar (sem key, rate limit, áudio corrupto),
retorna None. Vision_node então usa fallback genérico ("lead enviou áudio,
peça texto").
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
WHISPER_API_URL = os.getenv("WHISPER_API_URL", "https://api.openai.com/v1/audio/transcriptions")
WHISPER_TIMEOUT = float(os.getenv("WHISPER_TIMEOUT", "30"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "pt")  # ISO-639-1 — força pt-BR


# WhatsApp Baileys manda áudio em formato OGG/Opus geralmente. Whisper aceita
# vários formatos. Map mime → extensão (Whisper exige filename com ext válida).
_MIME_TO_EXT: dict[str, str] = {
    "audio/ogg": "ogg",
    "audio/oga": "ogg",
    "audio/opus": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/flac": "flac",
}


def _resolve_ext(mime: str) -> str:
    """Pega extensão do filename pra Whisper. Default ogg (WhatsApp default)."""
    if not mime:
        return "ogg"
    base = mime.split(";", 1)[0].strip().lower()
    return _MIME_TO_EXT.get(base, "ogg")


def _resolve_endpoint_and_key() -> tuple[str, str]:
    """
    Detecta provider + URL correta baseado em qual key está setada.

    Prioridade:
      1. WHISPER_API_URL env explícito (user override) + qualquer key
      2. OPENAI_API_KEY → api.openai.com/v1/audio/transcriptions (oficial)
      3. OPENROUTER_API_KEY → openrouter.ai/api/v1/audio/transcriptions
         (OpenRouter pode não suportar — fallback graceful)

    Returns: (url, api_key) ou ("", "") se nenhum key.
    """
    explicit_url = os.getenv("WHISPER_API_URL", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    router_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if openai_key:
        url = explicit_url or "https://api.openai.com/v1/audio/transcriptions"
        return (url, openai_key)
    if router_key:
        # OpenRouter pode proxy Whisper se mesmo path. Default endpoint OpenRouter.
        url = explicit_url or "https://openrouter.ai/api/v1/audio/transcriptions"
        return (url, router_key)
    return ("", "")


async def transcribe_audio(base64_data: str, mime: str) -> str | None:
    """
    Transcreve áudio base64 → texto via Whisper.

    Args:
        base64_data: bytes do áudio em base64 (Evolution webhookBase64=True)
        mime: ex "audio/ogg; codecs=opus"

    Returns:
        Texto transcrito ou None se falhou. Sempre retorna None safe (não levanta).
    """
    if not base64_data or not mime:
        return None

    api_url, api_key = _resolve_endpoint_and_key()
    if not api_key or not api_url:
        log.warning("[whisper] sem OPENAI_API_KEY/OPENROUTER_API_KEY — skip transcrição")
        return None

    try:
        audio_bytes = base64.b64decode(base64_data)
    except (ValueError, TypeError) as exc:
        log.warning("[whisper] base64 decode falhou: %s", exc)
        return None

    if len(audio_bytes) < 100:
        log.warning("[whisper] audio muito pequeno (%d bytes) — skip", len(audio_bytes))
        return None

    ext = _resolve_ext(mime)
    filename = f"audio.{ext}"
    mime_base = mime.split(";", 1)[0].strip().lower() or "audio/ogg"

    files = {"file": (filename, audio_bytes, mime_base)}
    data: dict[str, Any] = {
        "model": WHISPER_MODEL,
        "language": WHISPER_LANGUAGE,
        "response_format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=WHISPER_TIMEOUT) as client:
            r = await client.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
            )
            if r.status_code >= 400:
                log.warning(
                    "[whisper] HTTP %d: %s",
                    r.status_code, (r.text or "")[:200],
                )
                return None
            payload = r.json() if r.content else {}
            text = (payload.get("text") or "").strip()
            if not text:
                log.warning("[whisper] retornou vazio")
                return None
            log.info(
                "[whisper] transcrito %d bytes → %d chars: %s",
                len(audio_bytes), len(text), text[:80],
            )
            return text
    except httpx.HTTPError as exc:
        log.warning("[whisper] HTTPError: %s", exc)
        return None
    except (ValueError, KeyError) as exc:
        log.warning("[whisper] parse falhou: %s", exc)
        return None
