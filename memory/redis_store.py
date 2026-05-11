"""
Memória de curto prazo no Upstash Redis (REST) — histórico por cliente.

Mantém compatibilidade com o bot antigo:
  - Mesmo formato de chave: chat:{instance}_{phone}
  - Mesmo TTL de 72h
  - Dedup de messageId via SET NX
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


CHAT_TTL_SECONDS = 60 * 60 * 72        # 72h
DEDUP_TTL_SECONDS = 60 * 5             # 5 min
MAX_HISTORY = 200                       # mesma janela do bot antigo
MAX_CONTEXT_FOR_LLM = 24                # últimas N mensagens enviadas ao LLM


def _composite_id(instance: str, phone: str) -> str:
    return f"{instance}_{phone}"


def _chat_key(instance: str, phone: str) -> str:
    return f"chat:{_composite_id(instance, phone)}"


def _dedup_key(instance: str, phone: str, message_id: str) -> str:
    return f"msg_processed:{instance}:{phone}:{message_id}"


def _convo_summary_key(instance: str, phone: str) -> str:
    return f"summary:{_composite_id(instance, phone)}"


class RedisStore:
    """Wrapper assíncrono em torno do Upstash Redis REST API."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        timeout: float = 5.0,
    ):
        self.url = (url or os.getenv("UPSTASH_REDIS_REST_URL", "")).rstrip("/")
        self.token = token or os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        self.timeout = timeout
        self._fallback: dict[str, tuple[Any, float | None]] = {}

    @property
    def remote_enabled(self) -> bool:
        return bool(self.url and self.token)

    async def _cmd(self, *parts: str) -> Any:
        """
        Executa comando Redis via REST. Fallback in-memory se REST indisponível
        OU em caso de erro de rede/timeout — bot não pode derrubar webhook por flaky Redis.
        """
        if not self.remote_enabled:
            return self._local_cmd(parts)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=list(parts),
                )
                r.raise_for_status()
                return r.json().get("result")
        except (httpx.HTTPError, ValueError):
            # Network error, timeout, 5xx → degrade pra in-memory; perdemos persistência
            # entre processos mas mantemos webhook funcionando.
            return self._local_cmd(parts)

    def _local_cmd(self, parts: tuple[str, ...]) -> Any:
        cmd = parts[0].upper()
        now = time.time()

        if cmd == "GET":
            entry = self._fallback.get(parts[1])
            if not entry:
                return None
            value, exp = entry
            if exp is not None and exp <= now:
                self._fallback.pop(parts[1], None)
                return None
            return value

        if cmd == "SET":
            key, value = parts[1], parts[2]
            exp: float | None = None
            nx = False
            i = 3
            while i < len(parts):
                token = parts[i].upper()
                if token == "EX":
                    exp = now + float(parts[i + 1])
                    i += 2
                elif token == "NX":
                    nx = True
                    i += 1
                else:
                    i += 1
            if nx:
                # NX deve respeitar TTL: se chave expirou, deixa sobrescrever.
                existing = self._fallback.get(key)
                if existing is not None:
                    _, exp_existing = existing
                    if exp_existing is None or exp_existing > now:
                        return None  # chave viva → NX falha
                    # expirou → cai pra escrita
                    self._fallback.pop(key, None)
            self._fallback[key] = (value, exp)
            return "OK"

        if cmd == "DEL":
            return int(self._fallback.pop(parts[1], None) is not None)

        # ── LISTAS: LPUSH / RPUSH / RPOP / LPOP / LLEN ──
        # Armazena list como (list_obj, None) — segundo elemento (exp) ignorado.
        if cmd in ("LPUSH", "RPUSH"):
            key = parts[1]
            entry = self._fallback.get(key)
            if entry and isinstance(entry[0], list):
                lst = entry[0]
            else:
                lst = []
                self._fallback[key] = (lst, None)
            for val in parts[2:]:
                if cmd == "LPUSH":
                    lst.insert(0, val)
                else:
                    lst.append(val)
            return len(lst)

        if cmd in ("RPOP", "LPOP"):
            key = parts[1]
            entry = self._fallback.get(key)
            if not entry or not isinstance(entry[0], list) or not entry[0]:
                return None
            return entry[0].pop() if cmd == "RPOP" else entry[0].pop(0)

        if cmd == "LLEN":
            entry = self._fallback.get(parts[1])
            if not entry or not isinstance(entry[0], list):
                return 0
            return len(entry[0])

        return None

    # ────────────────────────────────────────────────────────────
    # Histórico de mensagens (formato compatível com o bot antigo)
    # ────────────────────────────────────────────────────────────

    async def append_message(
        self,
        instance: str,
        phone: str,
        role: str,           # "user" | "model"
        content: str,
    ) -> None:
        key = _chat_key(instance, phone)
        existing = await self._cmd("GET", key)
        history: list[dict] = []
        if existing:
            try:
                history = json.loads(existing) if isinstance(existing, str) else list(existing)
            except (json.JSONDecodeError, TypeError):
                history = []

        history.append({"role": role, "message": content, "ts": int(time.time())})
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        await self._cmd("SET", key, json.dumps(history), "EX", str(CHAT_TTL_SECONDS))

    async def load_history(
        self,
        instance: str,
        phone: str,
        limit: int = MAX_CONTEXT_FOR_LLM,
    ) -> list[BaseMessage]:
        """Lê histórico recente já como objetos LangChain."""
        key = _chat_key(instance, phone)
        existing = await self._cmd("GET", key)
        if not existing:
            return []
        try:
            history = json.loads(existing) if isinstance(existing, str) else list(existing)
        except (json.JSONDecodeError, TypeError):
            return []

        recent = history[-limit:] if limit > 0 else history
        out: list[BaseMessage] = []
        for item in recent:
            content = item.get("message", "")
            if item.get("role") == "model":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        return out

    async def mark_message_processed(
        self,
        instance: str,
        phone: str,
        message_id: str,
    ) -> bool:
        """SET NX para dedup. Retorna True se foi a primeira vez (deve processar)."""
        key = _dedup_key(instance, phone, message_id)
        result = await self._cmd("SET", key, "1", "NX", "EX", str(DEDUP_TTL_SECONDS))
        return result == "OK"

    # ────────────────────────────────────────────────────────────
    # Resumo opcional (sumarização longa) — chave separada
    # ────────────────────────────────────────────────────────────

    async def get_summary(self, instance: str, phone: str) -> str:
        return (await self._cmd("GET", _convo_summary_key(instance, phone))) or ""

    async def set_summary(self, instance: str, phone: str, summary: str) -> None:
        await self._cmd("SET", _convo_summary_key(instance, phone), summary, "EX", str(CHAT_TTL_SECONDS))

    # ────────────────────────────────────────────────────────────
    # KillSwitch follow-up — marca "lead" vs "agent" do último turno
    # ────────────────────────────────────────────────────────────

    async def set_last_from(self, instance: str, phone: str, who: str) -> None:
        """who = 'lead' | 'agent'. Usado pra cancelar follow-up se lead respondeu."""
        await self._cmd("SET", f"last_from:{instance}:{phone}", who, "EX", str(CHAT_TTL_SECONDS))

    async def get_last_from(self, instance: str, phone: str) -> str:
        return (await self._cmd("GET", f"last_from:{instance}:{phone}")) or ""

    # ────────────────────────────────────────────────────────────
    # Lock distribuído por instância (pra evitar race entre webhooks)
    # ────────────────────────────────────────────────────────────

    async def acquire_lock(self, instance: str, phone: str, ttl_seconds: int = 30) -> bool:
        """SET NX para lock. True = peguei. Caller deve liberar com release_lock()."""
        key = f"lock:{instance}:{phone}"
        result = await self._cmd("SET", key, "1", "NX", "EX", str(ttl_seconds))
        return result == "OK"

    async def release_lock(self, instance: str, phone: str) -> None:
        await self._cmd("DEL", f"lock:{instance}:{phone}")

    # ────────────────────────────────────────────────────────────
    # FIFO queue global — anti-spam WhatsApp
    # Garante 1 mensagem processada por vez (todos leads, todas instances).
    # Sem isso, paralelismo dispara N envios simultâneos → ban WhatsApp.
    # ────────────────────────────────────────────────────────────

    _QUEUE_KEY = "queue:messages"

    async def enqueue_message(self, payload: dict[str, Any]) -> int:
        """LPUSH na queue global. Retorna tamanho atual."""
        raw = json.dumps(payload)
        result = await self._cmd("LPUSH", self._QUEUE_KEY, raw)
        try:
            return int(result) if result is not None else 0
        except (TypeError, ValueError):
            return 0

    async def dequeue_message(self) -> dict[str, Any] | None:
        """RPOP FIFO. Retorna dict ou None se queue vazia."""
        raw = await self._cmd("RPOP", self._QUEUE_KEY)
        if not raw:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else None
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    async def queue_length(self) -> int:
        """LLEN — tamanho atual da queue."""
        try:
            res = await self._cmd("LLEN", self._QUEUE_KEY)
            return int(res or 0)
        except (TypeError, ValueError):
            return 0

    async def requeue_head(self, payload: dict[str, Any]) -> None:
        """RPUSH — coloca de volta NO FIM (próximo a sair). Usado em lock_held."""
        raw = json.dumps(payload)
        await self._cmd("RPUSH", self._QUEUE_KEY, raw)
