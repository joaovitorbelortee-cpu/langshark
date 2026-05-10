# Roadmap — bot-vendas (LangGraph WhatsApp sales bot)

**Milestone:** v1 — LangGraph maximalist
**Goal:** Mover **TUDO** que faz sentido pra dentro do StateGraph. Especialistas por intent, RAG, mídia, fluxos, follow-up, persistência durável, tool calls, streaming, testes.

## Phases

| #   | Nome                        | Status      | Goal                                                                                          |
|-----|-----------------------------|-------------|-----------------------------------------------------------------------------------------------|
| 1   | Postgres checkpointer       | pending     | AsyncPostgresSaver + InMemorySaver fallback. State durável, retomada automática.              |
| 2   | Vision node                 | pending     | vision_node injeta image_url/audio em HumanMessage; edge condicional pré-detect_intent.       |
| 3   | Especialistas (greet/objec/follow_up) | pending | Nós dedicados por intenção, prompts próprios, roteamento condicional.                  |
| 4   | Summarize node              | pending     | summarize_node quando histórico > 30 msgs. Resumo no Redis, contexto enxuto.                  |
| 5   | Flow executor               | pending     | flow_executor_node detecta `[FLOW: nome]` e dispara sequência via Evolution.                  |
| 6   | Tenant resolver (Supabase)  | pending     | tenant_resolver_node busca project_id em instance_projects.                                   |
| 7   | Evolution como @tool        | pending     | send_text/send_typing/send_reaction como `@tool` LangGraph + ToolNode.                        |
| 8   | Send node no grafo          | pending     | _send_chunks vira send_node interno; main.py só extrai e dispara.                             |
| 9   | Streaming                   | pending     | astream_events com logs ricos por nó.                                                          |
| 10  | Testes pytest               | pending     | End-to-end com mocks (LLM, Evolution, Redis).                                                 |

**Compat travada:**
- Endpoint `POST /webhook/evolution` + `POST /webhook`
- Header `apikey: $WEBHOOK_SECRET` (timing-safe)
- Chave Redis `chat:{instance}_{phone}`, TTL 72h
- Tags secretas `[COMPROU] [AGENDAR:N] [REACT:X] [QUOTE]`
