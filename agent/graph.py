"""
Construção do grafo LangGraph.

Topologia:

    START
      └─> load_history
            └─> detect_intent
                  ├─[comprou]──────────────> persist ─> END  (silencia follow-up)
                  ├─[intencao_compra]─────> retrieve_catalog ─> close_sale ─> persist ─> END
                  └─[outros]──────────────> retrieve_catalog ─> respond     ─> persist ─> END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    close_sale_node,
    detect_intent_node,
    load_history_node,
    persist_node,
    respond_node,
    retrieve_catalog_node,
)
from agent.state import SalesState


def _route_after_intent(state: SalesState) -> str:
    intent = state.get("intent", "outros")
    if intent == "comprou":
        return "persist"
    if intent == "intencao_compra":
        return "close_path"
    return "respond_path"


def build_graph():
    """Compila o grafo. Retorna um Runnable pronto pra ainvoke."""
    g: StateGraph = StateGraph(SalesState)

    g.add_node("load_history", load_history_node)
    g.add_node("detect_intent", detect_intent_node)
    g.add_node("retrieve_for_close", retrieve_catalog_node)
    g.add_node("retrieve_for_respond", retrieve_catalog_node)
    g.add_node("close_sale", close_sale_node)
    g.add_node("respond", respond_node)
    g.add_node("persist", persist_node)

    g.add_edge(START, "load_history")
    g.add_edge("load_history", "detect_intent")

    g.add_conditional_edges(
        "detect_intent",
        _route_after_intent,
        {
            "persist": "persist",
            "close_path": "retrieve_for_close",
            "respond_path": "retrieve_for_respond",
        },
    )

    g.add_edge("retrieve_for_close", "close_sale")
    g.add_edge("close_sale", "persist")

    g.add_edge("retrieve_for_respond", "respond")
    g.add_edge("respond", "persist")

    g.add_edge("persist", END)

    return g.compile()


_graph = None


def get_graph():
    """Singleton — compila uma vez por processo."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
