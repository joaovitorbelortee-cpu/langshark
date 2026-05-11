# Deploy — bot-vendas

Stack alvo: **Railway** (app), **Supabase** (Postgres + RLS), **Upstash Redis + QStash**, **Evolution API** (WhatsApp), **OpenRouter** (LLM).

---

## Pré-requisitos

| Conta | Plano mínimo | Pra que serve |
|-------|--------------|---------------|
| Railway | Free | Hospeda o webhook FastAPI |
| Supabase | Free | Postgres pra checkpointer + tabelas `products`/`instance_projects` |
| Upstash | Free | Redis (REST) + QStash (follow-up) |
| Evolution API | Self-hosted ou Railway | Bridge WhatsApp |
| OpenRouter | Pay-as-you-go | LLM hub |

CLIs (opcionais, recomendados):
```bash
npm i -g @railway/cli
npm i -g supabase
```

---

## 1) Supabase — Banco + tabelas

```bash
# Login
supabase login

# Link projeto existente (ou crie em https://supabase.com)
supabase link --project-ref <SEU_REF>

# Aplique migration de produtos + instance_projects
supabase db push
```

Alternativa (sem CLI): cole o conteúdo de `supabase/migrations/0001_products.sql` no SQL Editor do dashboard.

**Pegue (Settings → API):**
- `SUPABASE_URL` → `https://<ref>.supabase.co`
- `SUPABASE_SERVICE_KEY` (service_role) — bypassa RLS
- `POSTGRES_URL` (Settings → Database → Connection string → URI):
  `postgresql://postgres.<ref>:<senha>@aws-0-...:5432/postgres?sslmode=require`

**Popular catálogo (após upar produtos na tabela `products`):**
```bash
python -m scripts.sync_catalog --all
```

---

## 2) Upstash — Redis + QStash

1. Crie database em https://console.upstash.com/redis → pegue:
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`
2. Crie tópico QStash em https://console.upstash.com/qstash → pegue:
   - `QSTASH_TOKEN`
   - (opcional) `QSTASH_CURRENT_SIGNING_KEY` / `QSTASH_NEXT_SIGNING_KEY`

---

## 3) Evolution API

Já está rodando? Pegue:
- `EVOLUTION_API_URL` (ex: `https://evo.example.com`)
- `EVOLUTION_API_KEY` (admin key)
- `EVOLUTION_INSTANCE` (nome da instância, ex: `botzap`)

Se vai criar do zero: deploy do `evolution-api` (Docker oficial) num Railway próprio. Documentação: https://github.com/EvolutionAPI/evolution-api

---

## 4) Railway — deploy do bot-vendas

```bash
railway login
railway init                 # cria projeto novo
railway link                 # ou linka existente
railway up                   # build via Dockerfile (ou nixpacks fallback)
```

**Variáveis (Railway → Variables → Raw editor):**

```env
# Auth do webhook
WEBHOOK_SECRET=gere-um-secreto-forte-32-chars

# Evolution
EVOLUTION_API_URL=https://sua-evo.example.com
EVOLUTION_API_KEY=...
EVOLUTION_INSTANCE=botzap

# LLM
OPENROUTER_API_KEY=sk-or-...
AI_MODEL=openai/gpt-4o-mini
AI_BASE_URL=https://openrouter.ai/api/v1
AI_REFERRER=https://bot-vendas.up.railway.app

# Redis REST (queue/lock/lead_facts fallback)
UPSTASH_REDIS_REST_URL=https://...upstash.io
UPSTASH_REDIS_REST_TOKEN=...

# Redis TCP nativo (recommended — AsyncRedisSaver + AsyncRedisStore)
# Pega em Upstash → "Connect" tab (rediss://default:senha@host:6379)
# OU Railway Redis plugin, Redis Cloud, etc.
# Sem REDIS_URL, sistema usa POSTGRES_URL como checkpointer.
REDIS_URL=rediss://default:senha@us1-xxx.upstash.io:6379
STORE_TTL_DAYS=90   # TTL default no Store, refresh on read

# QStash
QSTASH_TOKEN=...
QSTASH_URL=https://qstash.upstash.io
PUBLIC_BASE_URL=https://bot-vendas.up.railway.app   # callback target

# Supabase
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_SERVICE_KEY=...
POSTGRES_URL=postgresql://postgres.<ref>:<senha>@aws-0-...:5432/postgres?sslmode=require

# Supervisor LLM (validador anti-burrice — pode desligar pra debug)
# SUPERVISOR_DISABLED=1
# SUPERVISOR_MODEL=openai/gpt-4o-mini
# SUPERVISOR_MAX_RETRIES=2

# Multi-tenant default
DEFAULT_PROJECT_ID=padrao

# RAG
CHROMA_DIR=/data/chroma     # já default no Dockerfile

# Opcional: ferramentas LangGraph autônomas
# ENABLE_TOOL_CALLS=1
```

**Volume persistente** (pra ChromaDB sobreviver a redeploys):
- Já declarado em `railway.toml` (`/data` mount em `bot-vendas-data`).
- Confira em Railway → Settings → Volumes que `bot-vendas-data` foi criado.

---

## 5) Configurar webhook na Evolution

```bash
curl -X POST "$EVOLUTION_API_URL/webhook/set/$EVOLUTION_INSTANCE" \
  -H "apikey: $EVOLUTION_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://bot-vendas.up.railway.app/webhook/evolution",
    "webhook_by_events": false,
    "webhook_base64": true,
    "events": ["MESSAGES_UPSERT", "CONNECTION_UPDATE", "PRESENCE_UPDATE"],
    "headers": {"apikey": "<MESMO_VALOR_DO_WEBHOOK_SECRET>"}
  }'
```

---

## 6) Validar — smoke test

```bash
# Localmente apontando pro deploy
python -m scripts.smoke --url https://bot-vendas.up.railway.app --secret <WEBHOOK_SECRET>
```

Esperado:
```
[ok] /health 200
[ok] /webhook 401 sem auth
[ok] /webhook 200 connection.update
[ok] /webhook 200 fromMe skip
[ok] /api/trigger-followup 400 sem instance
[PASS]: 0 failure(s)
```

---

## 7) Smoke real no WhatsApp

1. Mande "oi" pro número conectado na Evolution.
2. Logs Railway:
   - `[msg] padrao/botzap/55XXX: oi`
   - `[graph] > tenant_resolver` ... `> respond` ... `> send`
   - `[qstash] schedule_followup -> {ok: True, message_id: ...}`
3. Bot deve responder.
4. Espere o tempo de `[AGENDAR: N]` — QStash dispara `/api/trigger-followup` automaticamente.

---

## 8) Troubleshooting

| Sintoma | Causa | Fix |
|---------|-------|-----|
| `503 Server misconfiguration` no /webhook | `WEBHOOK_SECRET` ausente | Setar no Railway, `railway up` |
| `[checkpointer] usando InMemorySaver` em prod | `POSTGRES_URL` ausente/inválida | Verificar URI Supabase com `?sslmode=require` |
| LLM 401 / `incorrect api key` | `OPENROUTER_API_KEY` inválida | Regerar em openrouter.ai |
| `429 Rate limit` | OpenRouter free tier | Trocar `AI_MODEL` pra pago |
| Bot não responde | Evolution webhook não bate | Conferir `apikey` header bate com `WEBHOOK_SECRET` |
| Follow-up nunca dispara | QStash sem `PUBLIC_BASE_URL` | Setar URL pública (Railway exibe em Settings → Domains) |
| RAG vazio | Catálogo não sincronizado | `python -m scripts.sync_catalog --all` |

---

## 9) Logs em tempo real

```bash
railway logs --tail
```

Procurar:
- `[startup] grafo compilado (checkpointer=postgres)` — Postgres OK
- `[graph] >` `[graph] [ok]` — fluxo do nó por nó (streaming via astream_events)
- `[qstash] schedule_followup` — agendamento confirmado

---

## 10) Build local pra testar Docker antes de subir

```bash
docker build -t bot-vendas .
docker run --rm -p 8000:8000 \
  -e WEBHOOK_SECRET=test \
  -e EVOLUTION_API_URL=http://example.test \
  -e OPENROUTER_API_KEY=sk-or-... \
  bot-vendas

python -m scripts.smoke --url http://localhost:8000 --secret test
```

---

## Arquitetura final

```
WhatsApp -> Evolution API -> POST /webhook/evolution (Railway)
                                       |
                          [check_auth, dedup Redis, lock por phone]
                                       |
                                   LangGraph
                                       |
       tenant_resolver -> load_history -> summarize
                            |
              (mídia?) -> vision -> detect_intent
                            |
       saudacao -> greeting   ┐
       objecao  -> objection  ┤
       follow_up -> follow_up ┤
       intencao -> retrieve+close_sale ┤  -> [tools|flow|persist] -> send -> END
       comprou  ----------------------- ┤
       outros   -> retrieve+respond     ┘

         (state durável no Postgres Supabase via AsyncPostgresSaver)
         (histórico curto no Upstash Redis chat:{instance}_{phone})
         (RAG via ChromaDB persistente em /data/chroma)
         (follow-up agendado via QStash -> /api/trigger-followup)
```
