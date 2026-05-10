"""
Construção do grafo LangGraph.

Topologia (atualizada nas etapas seguintes):

    START
      └─> load_history
            └─> detect_intent
                  ├─[comprou]──────────────> persist ─> END
                  ├─[intencao_compra]─────> retrieve_catalog ─> close_sale ─> persist ─> END
                  └─[outros]──────────────> retrieve_catalog ─> respond     ─> persist ─> END

Compila com checkpointer opcional (Postgres ou in-memory) — passe `checkpointer`
em build_graph() pra ativar persistência durável de state entre invocações.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    close_sale_node,
    detect_intent_node,
    follow_up_node,
    greeting_node,
    load_history_node,
    objection_node,
    persist_node,
    respond_node,
    retrieve_catalog_node,
    summarize_node,
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


def build_graph(checkpointer: Any | None = None):
    """Compila o grafo. Retorna um Runnable pronto pra ainvoke."""
    g: StateGraph = StateGraph(SalesState)

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
    g.add_node("persist", persist_node)

    g.add_edge(START, "load_history")
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
    g.add_edge("close_sale", "persist")

    g.add_edge("retrieve_for_respond", "respond")
    g.add_edge("respond", "persist")

    g.add_edge("greeting", "persist")
    g.add_edge("objection", "persist")
    g.add_edge("follow_up", "persist")

    g.add_edge("persist", END)

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
