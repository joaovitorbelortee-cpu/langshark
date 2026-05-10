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
    greeting_node,
    load_history_node,
    objection_node,
    persist_node,
    respond_node,
    retrieve_catalog_node,
    send_node,
    summarize_node,
    tenant_resolver_node,
    vision_node,
)
from agent.state import SalesState


def _route_after_intent(state: SalesState) -> str:
    intent = state.get("intent", "outros")
    if intent == "comprou":
        return "persist"
    if intent == "saudacao":
        return "greeting"
    if intent == "objecao":
        return "objection"
    if intent == "follow_up":
        return "follow_up"
    if intent == "intencao_compra":
        return "close_path"
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
    """Depois de respond/close/specialists: tools > flow > persist."""
    if os.getenv("ENABLE_TOOL_CALLS") and _last_ai_has_tool_calls(state):
        return "tools_path"
    if state.get("flow_name"):
        return "flow_path"
    return "persist_path"


def build_graph(checkpointer: Any | None = None):
    """Compila o grafo. Retorna um Runnable pronto pra ainvoke."""
    g: StateGraph = StateGraph(SalesState)

    g.add_node("tenant_resolver", tenant_resolver_node)
    g.add_node("load_history", load_history_node)
    g.add_node("summarize", summarize_node)
    g.add_node("vision", vision_node)
    g.add_node("detect_intent", detect_intent_node)
    g.add_node("retrieve_for_close", retrieve_catalog_node)
    g.add_node("retrieve_for_respond", retrieve_catalog_node)
    g.add_node("close_sale", close_sale_node)
    g.add_node("respond", respond_node)
    g.add_node("greeting", greeting_node)
    g.add_node("objection", objection_node)
    g.add_node("follow_up", follow_up_node)
    g.add_node("flow_executor", flow_executor_node)
    g.add_node("tools", ToolNode(EVOLUTION_TOOLS))
    g.add_node("persist", persist_node)
    g.add_node("send", send_node)

    g.add_edge(START, "tenant_resolver")
    g.add_edge("tenant_resolver", "load_history")
    g.add_edge("load_history", "summarize")
    g.add_conditional_edges(
        "summarize",
        _route_after_history,
        {"vision_path": "vision", "intent_path": "detect_intent"},
    )
    g.add_edge("vision", "detect_intent")

    g.add_conditional_edges(
        "detect_intent",
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

    # Após qualquer nó que produz reply, decide entre tools / fluxo / persistência.
    for reply_node in ("close_sale", "respond", "greeting", "objection", "follow_up"):
        g.add_conditional_edges(
            reply_node,
            _route_after_reply,
            {
                "tools_path": "tools",
                "flow_path": "flow_executor",
                "persist_path": "persist",
            },
        )

    g.add_edge("tools", "persist")
    g.add_edge("flow_executor", "persist")
    g.add_edge("persist", "send")
    g.add_edge("send", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()


_graph: Any | None = None


def get_graph(checkpointer: Any | None = None):
    """Singleton — compila uma vez por processo (opcionalmente com checkpointer)."""
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=checkpointer)
    return _graph
