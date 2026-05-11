"""
Construção do grafo LangGraph.

Topologia:

    START
      └─> tenant_resolver
            └─> load_history
                  └─> summarize
                        └─[mídia?]─┬─> vision ─┐
                                   └──────────┬┘
                                              v
                                       detect_intent
                  ┌──────────[saudacao]───> greeting    ─┐
                  ├──────────[objecao]────> objection   ─┤
                  ├──────────[follow_up]──> follow_up   ─┤
                  ├──[intencao_compra]──> retrieve_for_close ─> close_sale ─┤
                  ├──────────[comprou]───────────────────────────────────────┤
                  └──────────[outros]──── retrieve_for_respond ─> respond  ─┤
                                                                            │
                                          ┌───────[flow_name?]──> flow_executor ─┤
                                          │                                      │
                                          v                                      v
                                       persist ─> send ─> END

Compila com checkpointer opcional (Postgres ou in-memory) — passe `checkpointer`
em build_graph() pra ativar persistência durável de state entre invocações.
"""
from __future__ import annotations

from typing import Any

import os

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.evolution_tools import EVOLUTION_TOOLS
from agent.nodes import (
    close_sale_node,
    detect_intent_node,
    flow_executor_node,
    follow_up_node,
    follow_up_strategist_node,
    greeting_node,
    lead_memory_node,
    load_history_node,
    load_system_prompt_node,
    objection_node,
    persist_node,
    respond_node,
    retrieve_catalog_node,
    send_node,
    summarize_node,
    supervisor_node,
    tenant_resolver_node,
    vision_node,
)
from agent.state import SalesState


def _route_after_intent(state: SalesState) -> str:
    intent = state.get("intent", "outros")
    if intent == "comprou":
        return "persist"
    # Verifica se bot já falou em algum turno passado → indica conversa em andamento
    msgs = state.get("messages") or []
    bot_already_spoke = any(getattr(m, "type", "") == "ai" for m in msgs)
    if intent == "saudacao":
        # Bot já falou? Conversa em andamento, NÃO saudação inicial → respond_path
        if bot_already_spoke:
            return "respond_path"
        return "greeting"
    if intent == "objecao":
        return "objection"
    if intent == "follow_up":
        # Follow-up MAS bot nunca falou? trata como saudação primeiro
        if not bot_already_spoke and not state.get("user_message"):
            return "greeting"
        return "follow_up"
    if intent == "intencao_compra":
        return "close_path"
    # "outros" + bot já falou = retomada/continuação, vai pra respond
    return "respond_path"


def _route_after_history(state: SalesState) -> str:
    """Se houver mídia base64, passa por vision antes do classificador."""
    if state.get("media_base64") and state.get("media_mime"):
        return "vision_path"
    return "intent_path"


def _last_ai_has_tool_calls(state: SalesState) -> bool:
    """Inspeciona última AIMessage do state.messages procurando tool_calls."""
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None)
            return bool(tool_calls)
    return False


def _route_after_reply(state: SalesState) -> str:
    """Depois de respond/close/specialists: tools > flow > supervisor."""
    if os.getenv("ENABLE_TOOL_CALLS") and _last_ai_has_tool_calls(state):
        return "tools_path"
    if state.get("flow_name"):
        return "flow_path"
    return "supervisor_path"


def _route_after_supervisor(state: SalesState) -> str:
    """
    Após supervisor avaliar:
      - rejected + tem retry pendente → volta pro respond_node com feedback
      - approved OU sem retries → strategist (segue normal)

    Safety: além de checar feedback, valida que supervisor_node de fato setou
    novos campos de retry (reply foi limpo). Se reply ainda existe = supervisor
    decidiu seguir (max retries OU approved). Evita loop infinito caso o
    supervisor_node esqueça de limpar feedback.
    """
    review = state.get("supervisor_review") or {}
    feedback = state.get("supervisor_feedback")
    reply = state.get("reply") or ""
    # Retry só rola se: reprovado + tem feedback + reply foi limpo pelo supervisor
    if not review.get("approved") and feedback and not reply.strip():
        return "retry_path"
    return "strategist_path"


def build_graph(checkpointer: Any | None = None, store: Any | None = None):
    """Compila o grafo. Retorna um Runnable pronto pra ainvoke.

    Args:
        checkpointer: thread-level state (BaseCheckpointSaver).
        store: cross-thread long-term memory (BaseStore). Opcional.
            Quando setado, nós podem usar via runtime.store.aput/aget/asearch.
    """
    g: StateGraph = StateGraph(SalesState)

    g.add_node("tenant_resolver", tenant_resolver_node)
    g.add_node("load_system_prompt", load_system_prompt_node)
    g.add_node("load_history", load_history_node)
    g.add_node("summarize", summarize_node)
    g.add_node("vision", vision_node)
    g.add_node("detect_intent", detect_intent_node)
    g.add_node("lead_memory", lead_memory_node)
    g.add_node("retrieve_for_close", retrieve_catalog_node)
    g.add_node("retrieve_for_respond", retrieve_catalog_node)
    g.add_node("close_sale", close_sale_node)
    g.add_node("respond", respond_node)
    g.add_node("greeting", greeting_node)
    g.add_node("objection", objection_node)
    g.add_node("follow_up", follow_up_node)
    g.add_node("flow_executor", flow_executor_node)
    g.add_node("tools", ToolNode(EVOLUTION_TOOLS))
    g.add_node("supervisor", supervisor_node)
    g.add_node("strategist", follow_up_strategist_node)
    g.add_node("persist", persist_node)
    g.add_node("send", send_node)

    g.add_edge(START, "tenant_resolver")
    g.add_edge("tenant_resolver", "load_system_prompt")
    g.add_edge("load_system_prompt", "load_history")
    g.add_edge("load_history", "summarize")
    g.add_conditional_edges(
        "summarize",
        _route_after_history,
        {"vision_path": "vision", "intent_path": "detect_intent"},
    )
    g.add_edge("vision", "detect_intent")

    # detect_intent → lead_memory (extrai fatos) → especialista
    g.add_edge("detect_intent", "lead_memory")
    g.add_conditional_edges(
        "lead_memory",
        _route_after_intent,
        {
            "persist": "persist",
            "greeting": "greeting",
            "objection": "objection",
            "follow_up": "follow_up",
            "close_path": "retrieve_for_close",
            "respond_path": "retrieve_for_respond",
        },
    )

    g.add_edge("retrieve_for_close", "close_sale")
    g.add_edge("retrieve_for_respond", "respond")

    # Após qualquer nó que produz reply: tools/flow/supervisor.
    # Supervisor valida ANTES de strategist+persist (anti-burrice).
    for reply_node in ("close_sale", "respond", "greeting", "objection", "follow_up"):
        g.add_conditional_edges(
            reply_node,
            _route_after_reply,
            {
                "tools_path": "tools",
                "flow_path": "flow_executor",
                "supervisor_path": "supervisor",
            },
        )

    g.add_edge("tools", "supervisor")
    g.add_edge("flow_executor", "supervisor")

    # Supervisor: aprovou → strategist; rejeitou → respond (retry com feedback)
    g.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "strategist_path": "strategist",
            "retry_path": "respond",
        },
    )

    g.add_edge("strategist", "persist")
    g.add_edge("persist", "send")
    g.add_edge("send", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store
    return g.compile(**compile_kwargs)


_graph: Any | None = None


def get_graph(checkpointer: Any | None = None, store: Any | None = None):
    """Singleton — compila uma vez por processo (opcionalmente com checkpointer/store)."""
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=checkpointer, store=store)
    return _graph
