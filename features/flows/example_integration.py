"""
Exemplo de integração — bot que reage a tag [FLOW: nome] no reply do LLM
disparando sequência cadastrada via canal arbitrário (WhatsApp/Telegram/etc).

Fluxo:
  1. LLM gera resposta com tag [FLOW: video_pitch]
  2. parse_flow_tag extrai → flow_name = "video_pitch"
  3. flow_executor.execute_flow dispara steps via sender
  4. Sender é seu cliente (Evolution/Telegram/Twilio) implementando MessageSender

Migrate Supabase primeiro:
    psql $DATABASE_URL -f features/flows/migrations/0001_flows_schema.sql
    psql $DATABASE_URL -f features/flows/migrations/0002_flow_media_bucket.sql
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

# Imports do pacote
from flows import (
    Flow,
    execute_flow,
    flows_prompt_block,
    get_flow,
    parse_flow_tag,
    register_flow,
)


# ─── Adapter pro seu canal — implemente MessageSender ────────────────
class WhatsAppSender:
    """Wrapper Evolution API (exemplo). Adapte pro Telegram/Twilio/etc."""

    def __init__(self, base_url: str, api_key: str, instance: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.instance = instance

    async def send_text(self, to: str, text: str) -> Any:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(
                f"{self.base_url}/message/sendText/{self.instance}",
                headers={"apikey": self.api_key, "Content-Type": "application/json"},
                json={"number": to, "text": text},
            )
            r.raise_for_status()
            return r.json()

    async def send_typing(self, to: str, duration_ms: int = 1200) -> Any:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as c:
            try:
                await c.post(
                    f"{self.base_url}/chat/sendPresence/{self.instance}",
                    headers={"apikey": self.api_key, "Content-Type": "application/json"},
                    json={"number": to, "presence": "composing", "delay": duration_ms},
                )
            except Exception:
                pass

    async def send_media(
        self,
        to: str,
        kind: str,
        url: str,
        caption: str | None = None,
        file_name: str | None = None,
    ) -> Any:
        import httpx
        body: dict[str, Any] = {
            "number": to,
            "mediatype": kind,
            "media": url,
            "caption": caption or "",
        }
        if kind == "document" and file_name:
            body["fileName"] = file_name
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(
                f"{self.base_url}/message/sendMedia/{self.instance}",
                headers={"apikey": self.api_key, "Content-Type": "application/json"},
                json=body,
            )
            r.raise_for_status()
            return r.json()


# ─── Setup: registra flow in-memory (ou usa Supabase via SUPABASE_URL) ──
PROJECT_ID = "padrao"

register_flow(PROJECT_ID, Flow(
    name="video_pitch",
    description="Cliente pediu pra ver demonstração ou material visual",
    steps=[
        {"type": "text", "content": "Vou te mandar um vídeo curto explicando 👇"},
        {"type": "video", "url": "https://cdn.example.com/pitch.mp4", "caption": "Veja isso"},
        {"type": "text", "content": "Curtiu? Quer mais detalhes de algum plano?"},
    ],
))


# ─── Uso: depois do LLM responder ────────────────────────────────────
async def handle_bot_reply(
    raw_reply: str,
    project_id: str,
    lead_phone: str,
    sender: WhatsAppSender,
) -> dict[str, Any]:
    """
    Processa reply do LLM. Se tem tag [FLOW: nome], dispara sequência.
    Senão, envia reply normal.
    """
    flow_name, cleaned_text = parse_flow_tag(raw_reply)

    # Sem tag → envia reply normal
    if not flow_name:
        await sender.send_text(lead_phone, cleaned_text)
        return {"mode": "text", "sent": cleaned_text}

    # Com tag → tenta executar flow
    flow = get_flow(project_id, flow_name)
    if not flow:
        # Flow não existe → fallback texto do LLM
        await sender.send_text(lead_phone, cleaned_text or "Vou te ajudar")
        return {"mode": "text_fallback", "reason": f"flow '{flow_name}' não encontrado"}

    result = await execute_flow(
        project_id=project_id,
        flow_name=flow_name,
        to=lead_phone,
        sender=sender,
    )

    # Fallback safety — se TODOS steps falharam, envia texto do LLM
    if not result["dispatched"] and cleaned_text:
        await sender.send_text(lead_phone, cleaned_text)
        result["fallback_sent"] = cleaned_text

    return {"mode": "flow", **result}


# ─── System prompt do LLM precisa do bloco de fluxos disponíveis ─────
def build_system_prompt(project_id: str, base_prompt: str) -> str:
    """Concatena seu prompt base + descrição dos flows pro LLM saber quais existem."""
    block = flows_prompt_block(project_id)
    if block:
        return f"{base_prompt}\n\n{block}"
    return base_prompt


# ─── Main exemplo ────────────────────────────────────────────────────
async def main():
    sender = WhatsAppSender(
        base_url=os.environ["EVOLUTION_API_URL"],
        api_key=os.environ["EVOLUTION_API_KEY"],
        instance=os.environ["EVOLUTION_INSTANCE"],
    )

    # Simula reply do LLM com tag
    fake_llm_reply = "Beleza, vou te mostrar! [FLOW: video_pitch]"
    result = await handle_bot_reply(
        raw_reply=fake_llm_reply,
        project_id=PROJECT_ID,
        lead_phone="5511999999999",
        sender=sender,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
