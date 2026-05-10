"""
Estado tipado do grafo de vendas.

Cada mensagem do WhatsApp instancia um SalesState que percorre os nós
(detect_intent → retrieve_catalog → respond → close → schedule_followup).
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


Intent = Literal[
    "saudacao",
    "duvida_produto",
    "pedir_preco",
    "objecao",
    "intencao_compra",
    "comprou",
    "follow_up",
    "outros",
]


class CatalogHit(TypedDict):
    """Item retornado pela busca RAG no catálogo."""
    id: str
    name: str
    description: str
    price: float | None
    score: float


class SalesState(TypedDict, total=False):
    # Identificação multi-tenant (mantém compat com instance_projects do schema antigo)
    project_id: str          # tenant
    instance_name: str       # instância Evolution
    phone: str               # telefone do lead (10–15 dígitos)
    push_name: str           # nome exibido no WhatsApp

    # Mensagem corrente
    user_message: str        # texto recebido (ou caption de mídia)
    media_mime: str | None   # mimetype quando há mídia
    media_base64: str | None

    # Histórico (LangGraph add_messages → append automático sem sobrescrever)
    messages: Annotated[list[BaseMessage], add_messages]

    # Roteamento
    intent: Intent
    catalog_hits: list[CatalogHit]   # produtos relevantes do RAG

    # Saída
    reply: str                       # resposta final ao cliente (já sem tags)
    chunks: list[str]                # bolhas para enviar ao WhatsApp
    has_converted: bool              # tag [COMPROU] detectada
    schedule_minutes: int | None     # tag [AGENDAR:N] detectada
    react_emoji: str | None          # tag [REACT:X] detectada
    quote_previous: bool             # tag [QUOTE] detectada

    # Memória longa (preenchida por summarize_node quando histórico cresce)
    summary: str                     # resumo das mensagens antigas

    # Fluxo pré-cadastrado detectado pela IA (tag [FLOW: nome])
    flow_name: str | None            # nome do fluxo a executar
    flow_dispatched: bool            # send foi feito via flow_executor_node

    # Tracing (preenchido pelo main durante streaming)
    message_id: str                  # id Evolution da mensagem do cliente

    # Resultado do envio (preenchido por send_node)
    sent: bool                       # se chunks foram entregues
    sent_count: int                  # quantas bolhas saíram com sucesso
