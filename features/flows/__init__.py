"""Flows — package portátil pra fluxos pré-cadastrados.

Public API:
    from flows import get_flow, list_flows, parse_flow_tag, execute_flow, Flow
"""
from .flow_executor import MessageSender, execute_flow
from .flows import (
    FLOW_REGISTRY,
    Flow,
    flows_prompt_block,
    get_flow,
    invalidate_flows_cache,
    list_flows,
    parse_flow_tag,
    register_flow,
)

__all__ = [
    "Flow",
    "FLOW_REGISTRY",
    "MessageSender",
    "execute_flow",
    "flows_prompt_block",
    "get_flow",
    "invalidate_flows_cache",
    "list_flows",
    "parse_flow_tag",
    "register_flow",
]
