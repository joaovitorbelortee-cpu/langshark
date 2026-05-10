"""Testes end-to-end do grafo LangGraph com mocks."""
from __future__ import annotations

import pytest

from agent.graph import build_graph
from agent.flows import Flow, register_flow


pytestmark = pytest.mark.asyncio


async def _run(graph, **state_kwargs):
    initial = {
        "project_id": "padrao",
        "instance_name": "botzap",
        "phone": "5511999999999",
        "push_name": "Test",
        "user_message": "oi",
        "media_mime": None,
        "media_base64": None,
        "message_id": "msg123",
        "messages": [],
        **state_kwargs,
    }
    return await graph.ainvoke(initial)


async def test_graph_greeting_path(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    """saudacao → greeting_node → persist → send"""
    fake_llm(["saudacao", "Oi! Tudo bem? [AGENDAR: 20]"])
    graph = build_graph()

    final = await _run(graph, user_message="boa tarde")

    assert final["intent"] == "saudacao"
    assert "Oi!" in final["reply"]
    assert final["schedule_minutes"] == 20
    assert len(fake_evolution.sent_text) >= 1
    assert "Oi!" in fake_evolution.sent_text[0][2]


async def test_graph_close_path(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    """intencao_compra → retrieve_for_close → close_sale → persist → send"""
    fake_llm(["intencao_compra", "Fechado! Te mando o pix em 1min [AGENDAR: 15]"])
    graph = build_graph()

    final = await _run(graph, user_message="quero comprar")

    assert final["intent"] == "intencao_compra"
    assert final["schedule_minutes"] == 15
    assert len(fake_evolution.sent_text) >= 1


async def test_graph_comprou_skips_specialist(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    """comprou → persist direto (sem chamar especialista)"""
    fake_llm(["comprou"])  # só uma chamada — detect_intent
    graph = build_graph()

    final = await _run(graph, user_message="acabei de pagar")

    assert final["intent"] == "comprou"
    # Não houve respond/close/specialist → sem chunks → send pula com sent_count=0
    assert final.get("sent_count", 0) == 0
    assert len(fake_evolution.sent_text) == 0


async def test_graph_objection_path(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    fake_llm(["objecao", "Entendo perfeitamente. E se eu te mostrar o ROI? [AGENDAR: 60]"])
    graph = build_graph()

    final = await _run(graph, user_message="ta caro demais")

    assert final["intent"] == "objecao"
    assert final["schedule_minutes"] == 60
    assert len(fake_evolution.sent_text) >= 1


async def test_graph_react_emoji_sent(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    fake_llm(["saudacao", "Eai! [REACT:👋] [AGENDAR: 20]"])
    graph = build_graph()

    await _run(graph, user_message="oi")

    assert len(fake_evolution.sent_reaction) == 1
    inst, to, mid, emoji = fake_evolution.sent_reaction[0]
    assert emoji == "👋"
    assert mid == "msg123"


async def test_graph_vision_path(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    """media_base64 → vision_node injeta multimodal HumanMessage antes do detect."""
    fake_llm(["outros", "vi sua imagem [AGENDAR: 30]"])
    graph = build_graph()

    final = await _run(
        graph,
        user_message="o que acha disso?",
        media_mime="image/jpeg",
        media_base64="ZmFrZQ==",  # 'fake' em base64
    )

    assert final["intent"] == "outros"
    assert len(fake_evolution.sent_text) >= 1


async def test_graph_flow_executor_diverts_send(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    """Tag [FLOW: nome] dispara fluxo cadastrado e pula bolhas padrão."""
    register_flow(
        "padrao",
        Flow(
            name="boas_vindas",
            steps=[
                {"type": "text", "content": "Bem-vindo!"},
                {"type": "text", "content": "Como posso ajudar?"},
            ],
        ),
    )
    fake_llm(["saudacao", "Manda fluxo [FLOW: boas_vindas] [AGENDAR: 20]"])
    graph = build_graph()

    final = await _run(graph, user_message="oi")

    assert final.get("flow_dispatched") is True
    # send_node pula porque flow_dispatched=True (mas o flow_executor já enviou)
    sent_msgs = [t[2] for t in fake_evolution.sent_text]
    assert "Bem-vindo!" in sent_msgs
    assert "Como posso ajudar?" in sent_msgs


async def test_graph_tenant_resolver_fills_project_id(
    fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant
):
    fake_llm(["saudacao", "oi [AGENDAR: 20]"])
    graph = build_graph()

    final = await _run(graph, project_id="", instance_name="botzap")

    assert final["project_id"] == "padrao"  # mapeado por FakeTenant


async def test_graph_persist_writes_redis(fake_llm, fake_evolution, fake_redis, fake_rag, fake_tenant):
    fake_llm(["saudacao", "Oi! [AGENDAR: 20]"])
    graph = build_graph()

    await _run(graph, user_message="boa tarde")

    history = await fake_redis.load_history("botzap", "5511999999999")
    contents = [m.content for m in history if isinstance(m.content, str)]
    assert "boa tarde" in contents
    assert any("Oi!" in c for c in contents)
