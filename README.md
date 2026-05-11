# 🦈 LangShark

**WhatsApp sales bot built with LangGraph.** Multi-agent specialists, persistent state, RAG, follow-up scheduling — production-ready on Railway.

[![tests](https://img.shields.io/badge/tests-28%20passing-success)]()
[![python](https://img.shields.io/badge/python-3.11+-blue)]()
[![langgraph](https://img.shields.io/badge/langgraph-0.2+-purple)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()

---

## ✨ Features

- **LangGraph state machine** — every turn flows through typed nodes (intent detection, RAG, specialists, send)
- **Multi-specialist** — `greeting` / `objection` / `follow_up` / `close_sale` / `respond` — each inheriting a single `SALES_SYSTEM` and adding only a focus delta
- **Vision-ready** — image/audio/video routed through a `vision_node` that injects multimodal `HumanMessage`
- **Persistent state** — `AsyncPostgresSaver` (Supabase Postgres) checkpointing every conversation; restart-safe
- **Long memory** — `summarize_node` trims history with `RemoveMessage` once it crosses 30 messages
- **RAG** — products fetched from Supabase + ranked via Jaccard overlap (no ChromaDB needed in prod)
- **Follow-up scheduler** — `[AGENDAR: N]` tag → QStash schedules `/api/trigger-followup` callback; KillSwitch cancels if lead replies first
- **Per-phone serialization** — Redis lock prevents concurrent graph runs on the same conversation
- **Streaming logs** — `astream_events` traces every node entry/exit
- **Multi-tenant** — `instance_projects` table maps Evolution instance → project_id
- **Tag-based control** — `[COMPROU]`, `[AGENDAR:N]`, `[REACT:emoji]`, `[QUOTE]`, `[FLOW:nome]`
- **Tool-use ready** — `EvolutionClient` exposed as `@tool` LangGraph; `bind_tools(...)` opt-in via `ENABLE_TOOL_CALLS=1`
- **Resilient** — exponential retry on Evolution 5xx/429/timeout; `slowapi` rate limit on webhook

---

## 🧠 Architecture

```
WhatsApp ─► Evolution API ─► POST /webhook/evolution
                                     │
                       [auth, dedup, lock]
                                     ▼
                               LangGraph
                                     │
   tenant_resolver ─► load_system_prompt ─► load_history ─► summarize
                                                                 │
                              (mídia?) ─► vision ─► detect_intent
                                                          │
       saudacao  ─► greeting    ┐
       objecao   ─► objection   ┤
       follow_up ─► follow_up   ┤
       intencao  ─► retrieve+close_sale  ┤  ─► [tools|flow|persist] ─► send ─► END
       comprou   ──────────────────────  ┤
       outros    ─► retrieve+respond     ┘
```

State (Postgres-backed via `AsyncPostgresSaver`)
- Short-term memory: Upstash Redis `chat:{instance}_{phone}` (72h TTL)
- Catalog RAG: Supabase `products` table
- Tenant: Supabase `instance_projects` table
- Follow-up: QStash schedules webhook callback after `[AGENDAR: N]` minutes

---

## 🚀 Quick start (local)

```bash
git clone https://github.com/joaovitorbelortee-cpu/langshark.git
cd langshark

python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows
# source .venv/bin/activate              # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# fill: WEBHOOK_SECRET, EVOLUTION_API_*, OPENROUTER_API_KEY, UPSTASH_*, SUPABASE_*

python -m rag.seed_example padrao        # populate ChromaDB (local dev)
pytest -q                                # 28 tests
python main.py                           # http://localhost:8000
```

---

## 🌐 Deploy (Railway + Supabase + Upstash)

See [DEPLOY.md](./DEPLOY.md) for full step-by-step.

```bash
# 1. Supabase migration
psql $POSTGRES_URL -f supabase/migrations/0001_products.sql

# 2. Railway
railway login
railway init --name langshark
railway up
railway domain

# 3. Set vars (use scripts/railway_setvars.ps1 for batch)

# 4. Configure Evolution webhook
curl -X POST "$EVOLUTION_API_URL/webhook/set/$EVOLUTION_INSTANCE" \
  -H "apikey: $EVOLUTION_API_KEY" \
  -d '{
    "webhook": {
      "enabled": true,
      "url": "https://<your-app>.up.railway.app/webhook/evolution",
      "headers": {"apikey": "<WEBHOOK_SECRET>"},
      "events": ["MESSAGES_UPSERT", "CONNECTION_UPDATE", "PRESENCE_UPDATE"]
    }
  }'

# 5. Smoke
python -m scripts.smoke --url https://<your-app>.up.railway.app --secret <WEBHOOK_SECRET>
```

---

## 📁 Structure

```
langshark/
├── main.py                          # FastAPI webhook (slim — auth/dedup/lock only)
├── agent/
│   ├── state.py                     # SalesState TypedDict + Intent literals
│   ├── graph.py                     # StateGraph topology
│   ├── nodes.py                     # 16 nodes + SALES_SYSTEM + specialist focuses
│   ├── tools.py                     # parse_tags, chunk_for_whatsapp, EvolutionClient (with retry)
│   ├── flows.py                     # [FLOW:nome] tag handler + registry
│   ├── evolution_tools.py           # @tool wrappers for ToolNode
│   ├── checkpointer.py              # AsyncPostgresSaver + InMemorySaver fallback
│   └── qstash.py                    # follow-up scheduler client
├── memory/
│   ├── redis_store.py               # Upstash REST + lock + KillSwitch
│   └── supabase_tenant.py           # instance_projects → project_id
├── rag/
│   ├── catalog.py                   # ChromaDB (local dev)
│   ├── supabase_rag.py              # Supabase RAG (prod)
│   └── seed_example.py              # demo seed
├── scripts/
│   ├── smoke.py                     # 5 e2e checks against live deploy
│   ├── sync_catalog.py              # Supabase products → ChromaDB sync
│   └── railway_setvars.ps1          # batch env vars setup
├── supabase/
│   └── migrations/
│       └── 0001_products.sql        # products + instance_projects + RLS
├── tests/                           # 28 passing
│   ├── conftest.py                  # mocks LLM/Evolution/Redis/RAG/Tenant
│   ├── test_tools.py                # tag parsing, chunking, flow detection
│   ├── test_graph.py                # 9 e2e graph paths
│   ├── test_retry.py                # backoff 5xx/429/4xx/timeout
│   └── test_checkpointer.py         # InMemory + skip-if-no-postgres live
├── Dockerfile
├── railway.toml
├── nixpacks.toml
├── pytest.ini
├── requirements.txt
├── DEPLOY.md
└── README.md
```

---

## 🎯 Tag protocol

The bot emits these tags inside the LLM reply; `parse_tags` strips them before sending to WhatsApp.

| Tag | Meaning | Effect |
|-----|---------|--------|
| `[COMPROU]` | Customer paid | Disables follow-ups |
| `[AGENDAR: N]` | Schedule next contact in N minutes (5–10080) | QStash callback |
| `[REACT: 🔥]` | React to last customer message | `sendReaction` Evolution endpoint |
| `[QUOTE]` | Reply citing previous message | WhatsApp quote feature |
| `[FLOW: nome]` | Trigger a registered media sequence | `flow_executor_node` dispatches |

---

## 🧪 Testing

```bash
pytest -q                         # 28 unit + e2e (mocks, no network)
python -m scripts.smoke --url https://<deploy>  # 5 live checks
```

Mocks (`tests/conftest.py`): `FakeLLM` (scripted), `FakeEvolution` (capture), `FakeRedis` (in-memory), `FakeRAG`, `FakeTenant`. Zero network in unit tests.

---

## 🔧 Configuration

All via environment variables. See [`.env.example`](./.env.example).

Critical:

| Var | Required | Purpose |
|-----|----------|---------|
| `WEBHOOK_SECRET` | ✓ | Timing-safe auth on `/webhook/evolution` |
| `EVOLUTION_API_URL` + `_API_KEY` + `_INSTANCE` | ✓ | WhatsApp bridge |
| `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) | ✓ | LLM |
| `UPSTASH_REDIS_REST_URL` + `_TOKEN` | ✓ | Short memory + lock |
| `SUPABASE_URL` + `_SERVICE_KEY` | ✓ | Multi-tenant + RAG |
| `POSTGRES_URL` | recommended | Durable checkpointer (else InMemory) |
| `QSTASH_TOKEN` + `PUBLIC_BASE_URL` | recommended | Follow-up scheduler |
| `ENABLE_TOOL_CALLS` | optional | Bind `@tool` evolution funcs to LLM |

---

## 🤝 Why "LangShark"?

Built on **Lang**Graph. Mean like a **Shark** — closes deals fast, doesn't let leads escape.

---

## 📜 License

MIT — see [LICENSE](./LICENSE).
