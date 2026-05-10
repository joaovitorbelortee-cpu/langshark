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
        """Executa um comando Redis via REST. Cai pra fallback in-memory se faltar config."""
        if not self.remote_enabled:
            return self._local_cmd(parts)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                json=list(parts),
            )
            r.raise_for_status()
            return r.json().get("result")

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
            if nx and parts[1] in self._fallback:
                return None
            self._fallback[key] = (value, exp)
            return "OK"

        if cmd == "DEL":
            return int(self._fallback.pop(parts[1], None) is not None)

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
