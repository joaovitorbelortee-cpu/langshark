"""
Ferramentas LangChain (@tool) que encapsulam o EvolutionClient.

Servem em dois cenários:

1. **ToolNode reativo** — usado quando a IA decide CHAMAR uma ferramenta
   (ex.: reagir com emoji, marcar como lido). A IA emite tool_calls e o
   ToolNode despacha.

2. **Chamada direta pelo send_node** — funções utilitárias que mandam texto e
   mídia sem passar pela IA (envio determinístico das bolhas).

Note: como o EvolutionClient é singleton injetável (via agent.nodes.set_evolution),
as @tools delegam pro mesmo cliente — facilita mock em testes.
"""
from __future__ import annotations

from langchain_core.tools import tool

from agent.nodes import get_evolution


@tool("send_whatsapp_text", parse_docstring=True)
async def send_whatsapp_text(instance: str, to: str, text: str) -> dict:
    """Envia uma mensagem de texto no WhatsApp via Evolution API.

    Args:
        instance: Nome da instância Evolution (ex: 'botzap').
        to: Telefone do destinatário, só dígitos (10–15 caracteres).
        text: Conteúdo da mensagem.
    """
    return await get_evolution().send_text(instance, to, text)


@tool("send_whatsapp_typing", parse_docstring=True)
async def send_whatsapp_typing(instance: str, to: str, duration_ms: int = 2000) -> dict:
    """Mostra 'digitando...' no WhatsApp do destinatário.

    Args:
        instance: Nome da instância Evolution.
        to: Telefone do destinatário.
        duration_ms: Duração em milissegundos (default 2000).
    """
    return await get_evolution().send_typing(instance, to, duration_ms=duration_ms)


@tool("react_to_message", parse_docstring=True)
async def react_to_message(instance: str, to: str, message_id: str, emoji: str) -> dict:
    """Reage com um emoji a uma mensagem específica do cliente.

    Args:
        instance: Nome da instância Evolution.
        to: Telefone do destinatário.
        message_id: ID Evolution da mensagem original do cliente.
        emoji: Emoji para reação (ex: '👍', '❤️').
    """
    return await get_evolution().send_reaction(instance, to, message_id, emoji)


@tool("mark_message_read", parse_docstring=True)
async def mark_message_read(instance: str, remote_jid: str, message_id: str) -> dict:
    """Marca uma mensagem do cliente como lida (✓✓ azul no WhatsApp).

    Args:
        instance: Nome da instância Evolution.
        remote_jid: JID Evolution (ex: '5511999999999@s.whatsapp.net').
        message_id: ID Evolution da mensagem.
    """
    return await get_evolution().mark_read(instance, remote_jid, message_id)


# Lista pronta para passar em bind_tools(...) ou ToolNode([...])
EVOLUTION_TOOLS = [
    send_whatsapp_text,
    send_whatsapp_typing,
    react_to_message,
    mark_message_read,
]
