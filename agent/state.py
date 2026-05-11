"""
Estado tipado do grafo de vendas.

Cada mensagem do WhatsApp instancia um SalesState que percorre os nós
(detect_intent → retrieve_catalog → respond → close → schedule_followup).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, TypedDict

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


class SalesState(TypedDict):
    """
    Estado completo do grafo. `NotRequired[]` marca campos opcionais explicitamente
    (Python 3.11+). Campos sem NotRequired devem estar presentes no initial_state.
    """
    # Identificação multi-tenant (mantém compat com instance_projects do schema antigo)
    project_id: str                    # tenant — sempre presente após tenant_resolver
    instance_name: str                 # instância Evolution
    phone: str                         # telefone do lead (10–15 dígitos)
    push_name: NotRequired[str]        # nome exibido no WhatsApp

    # Mensagem corrente
    user_message: str                  # texto recebido (ou caption de mídia)
    media_mime: NotRequired[str | None]
    media_base64: NotRequired[str | None]

    # Histórico (LangGraph add_messages → append automático sem sobrescrever)
    messages: Annotated[list[BaseMessage], add_messages]

    # Roteamento
    intent: NotRequired[Intent]
    catalog_hits: NotRequired[list[CatalogHit]]

    # Saída
    reply: NotRequired[str]
    chunks: NotRequired[list[str]]
    has_converted: NotRequired[bool]
    schedule_minutes: NotRequired[int | None]
    react_emoji: NotRequired[str | None]
    quote_previous: NotRequired[bool]

    # System prompt principal (carregado por load_system_prompt_node)
    system_prompt: NotRequired[str]

    # Memória longa
    summary: NotRequired[str]

    # Fluxo pré-cadastrado (tag [FLOW: nome])
    flow_name: NotRequired[str | None]
    flow_dispatched: NotRequired[bool]

    # Tracing
    message_id: NotRequired[str]

    # Resultado do envio
    sent: NotRequired[bool]
    sent_count: NotRequired[int]

    # Follow-up Strategist (preenchido por follow_up_strategist_node)
    follow_up_strategy: NotRequired[dict[str, Any]]   # {temperatura, razao, abordagem, ...}
    follow_up_attempts: NotRequired[int]              # nº de followups já enviados sem resposta

    # Lead Memory (preenchido por lead_memory_node) — anti-amnésia
    lead_facts: NotRequired[dict[str, Any]]           # {plataforma, nome, plano_interesse, estagio, ...}

    # Supervisor (preenchido por supervisor_node) — validação anti-burrice
    supervisor_review: NotRequired[dict[str, Any]]    # {approved, reason, feedback, severity}
    supervisor_attempts: NotRequired[int]             # nº de retries já tentados
    supervisor_feedback: NotRequired[str]             # feedback do supervisor pra retry do especialista

    # Debug — quando True, send_node não envia pra Evolution (modo simulate)
    _debug_no_send: NotRequired[bool]
