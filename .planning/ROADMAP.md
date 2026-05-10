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

---

## Milestone v2 — Production Hardening (Railway + Supabase + QStash)

**Goal:** Fechar gaps de produção. Deploy 100% no Railway com Supabase Postgres e QStash follow-up.

| #   | Nome                              | Status   | Goal                                                                                |
|-----|-----------------------------------|----------|-------------------------------------------------------------------------------------|
| 11  | Summarize trim com RemoveMessage  | pending  | Corrigir leak de histórico no checkpointer com RemoveMessage reducer.               |
| 12  | QStash follow-up scheduler        | pending  | `/api/trigger-followup` callback + agendamento usando `schedule_minutes`.           |
| 13  | Rate limit + retry Evolution      | pending  | slowapi no webhook + tenacity backoff no EvolutionClient.                           |
| 14  | Fila Redis por instância          | pending  | ZADD + worker lock (compat com bot antigo). Evita race por phone.                   |
| 15  | Importer catálogo Supabase→Chroma | pending  | Script `scripts/sync_catalog.py` + tabela `products` no Supabase.                   |
| 16  | bind_tools real no respond_node   | pending  | ToolNode ativado. IA decide quando reagir, marcar lido.                             |
| 17  | Deploy configs                    | pending  | Dockerfile + railway.toml + nixpacks.toml + Procfile + Supabase migration SQL.      |
| 18  | Postgres checkpointer real        | pending  | Validar AsyncPostgresSaver com Supabase Postgres (URL ?sslmode=require).            |
| 19  | Smoke test integração             | pending  | Script `scripts/smoke.py` que valida health, fake webhook, grafo end-to-end.        |
| 20  | Deploy checklist + docs           | pending  | DEPLOY.md com comandos exatos Railway/Supabase/QStash + variáveis necessárias.      |
