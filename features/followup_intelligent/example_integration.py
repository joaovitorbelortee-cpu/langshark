"""
Exemplo de integração — FastAPI app que recebe msg de cliente, decide follow-up,
agenda no QStash. Quando QStash dispara, endpoint /api/trigger-followup envia
mensagem de reengajamento.

Adapte pro seu canal (WhatsApp/Telegram/SMS). A parte que importa:
1. Após cada turn do bot, chamar classify_lead
2. Se decision.agendar_minutos > 0 e não killswitch → qstash.schedule_followup
3. Quando lead RESPONDE, marcar last_message_from=lead (KillSwitch)
4. Endpoint /api/trigger-followup checa KillSwitch antes de enviar
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

# Imports do pacote
from followup_intelligent import classify_lead, QStashClient


app = FastAPI()
qstash = QStashClient()  # lê QSTASH_TOKEN + PUBLIC_BASE_URL + WEBHOOK_SECRET de env


# ─── Storage abstrato: substitua por Redis/Postgres ──────────────────
class FollowupStore:
    """Stub — substitua por Redis em produção."""
    _attempts: dict[str, int] = {}
    _last_from: dict[str, str] = {}  # "lead" ou "agent"
    _conversations: dict[str, list] = {}

    @classmethod
    async def get_attempts(cls, lead_id: str) -> int:
        return cls._attempts.get(lead_id, 0)

    @classmethod
    async def increment_attempts(cls, lead_id: str) -> int:
        cls._attempts[lead_id] = cls._attempts.get(lead_id, 0) + 1
        return cls._attempts[lead_id]

    @classmethod
    async def reset_attempts(cls, lead_id: str) -> None:
        cls._attempts.pop(lead_id, None)

    @classmethod
    async def set_last_from(cls, lead_id: str, who: str) -> None:
        cls._last_from[lead_id] = who

    @classmethod
    async def get_last_from(cls, lead_id: str) -> str | None:
        return cls._last_from.get(lead_id)

    @classmethod
    async def append_message(cls, lead_id: str, msg) -> None:
        cls._conversations.setdefault(lead_id, []).append(msg)

    @classmethod
    async def get_messages(cls, lead_id: str) -> list:
        return cls._conversations.get(lead_id, [])


store = FollowupStore()


# ─── Endpoint do canal (lead manda msg) ──────────────────────────────
@app.post("/webhook/lead-message")
async def handle_lead_message(req: Request) -> dict:
    """Recebe msg do lead, bot responde, agenda follow-up se aplicável."""
    body = await req.json()
    lead_id = body["lead_id"]
    user_msg = body["text"]

    # 1. Marca KillSwitch — lead falou
    await store.set_last_from(lead_id, "lead")
    await store.reset_attempts(lead_id)  # zera tentativas, lead ativo

    # 2. Persist user msg
    await store.append_message(lead_id, HumanMessage(content=user_msg))

    # 3. Bot responde (substitua pela sua lógica)
    bot_reply = your_bot_logic(user_msg)
    await store.append_message(lead_id, AIMessage(content=bot_reply))
    await store.set_last_from(lead_id, "agent")
    # ... envia bot_reply pelo canal (WhatsApp/Telegram/etc)

    # 4. Decide follow-up
    attempts = await store.get_attempts(lead_id)
    messages = await store.get_messages(lead_id)
    decision = await classify_lead(messages, user_msg, attempts)

    # 5. Agenda se faz sentido
    if decision["agendar_minutos"] > 0 and not decision["killswitch_permanent"]:
        result = await qstash.schedule_followup(
            delay_minutes=decision["agendar_minutos"],
            payload={
                "lead_id": lead_id,
                "abordagem": decision["abordagem"],
                "temperatura": decision["temperatura"],
            },
        )
        return {"ok": True, "decision": decision, "schedule": result}

    return {"ok": True, "decision": decision, "schedule": None}


# ─── Endpoint que QStash dispara ─────────────────────────────────────
@app.post("/api/trigger-followup")
async def trigger_followup(req: Request) -> dict:
    """Roda quando QStash dispara após delay."""
    # 1. Auth (QStash forward o WEBHOOK_SECRET como apikey)
    apikey = req.headers.get("apikey", "")
    if apikey != os.getenv("WEBHOOK_SECRET", ""):
        raise HTTPException(401, "Unauthorized")

    body = await req.json()
    lead_id = body["lead_id"]

    # 2. KillSwitch: lead respondeu? não envia.
    last_from = await store.get_last_from(lead_id)
    if last_from == "lead":
        return {"ok": True, "skipped": "lead_replied"}

    # 3. Incrementa tentativas
    attempts = await store.increment_attempts(lead_id)

    # 4. Gera msg de reengajamento (sua lógica)
    abordagem = body.get("abordagem", "valor")
    followup_msg = generate_followup_message(lead_id, abordagem, attempts)

    # 5. Envia via canal (WhatsApp/Telegram/etc) — adapte
    # await whatsapp.send_text(lead_id, followup_msg)

    # 6. Marca bot como último a falar
    await store.set_last_from(lead_id, "agent")
    await store.append_message(lead_id, AIMessage(content=followup_msg))

    # 7. Re-classifica pra próximo follow-up
    messages = await store.get_messages(lead_id)
    decision = await classify_lead(messages, "", attempts)
    if decision["agendar_minutos"] > 0 and not decision["killswitch_permanent"]:
        await qstash.schedule_followup(
            delay_minutes=decision["agendar_minutos"],
            payload={
                "lead_id": lead_id,
                "abordagem": decision["abordagem"],
                "temperatura": decision["temperatura"],
            },
        )

    return {"ok": True, "sent": followup_msg, "attempts": attempts}


def your_bot_logic(user_msg: str) -> str:
    """Substitua pela sua lógica de bot (LLM, regras, etc)."""
    return "Beleza, vou te ajudar"


def generate_followup_message(lead_id: str, abordagem: str, attempts: int) -> str:
    """Gera msg de reengajamento por abordagem Cialdini."""
    if abordagem == "commitment":
        return "Oi! Lembrei que você comentou que ia ver com calma. Conseguiu pensar?"
    if abordagem == "escassez":
        return "Oi, só pra avisar: restam poucas vagas dessa semana."
    if abordagem == "reciprocidade":
        return "Te separei uma dica que pode interessar..."
    if abordagem == "social":
        return "Acabou de fechar mais um cliente igual seu caso. Quer ver?"
    return "Oi, tudo certo? Posso ajudar com mais alguma coisa?"
