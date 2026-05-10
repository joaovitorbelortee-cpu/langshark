# State

**Milestones:** v1 (LangGraph) + v2 (Production Hardening) — todos completos
**Status:** READY-FOR-DEPLOY (Railway + Supabase + QStash)

## Progress v1

| Phase | Status    | Commit   | What                                                          |
|-------|-----------|----------|---------------------------------------------------------------|
| 1     | completed | f2f7e3f  | Postgres/InMemory checkpointer + build_graph(checkpointer=)   |
| 2     | completed | 3c0cc3d  | vision_node multimodal + conditional edge                     |
| 3     | completed | 06d6e65  | greeting/objection/follow_up specialists                      |
| 4     | completed | f6c8bd0  | summarize_node + Redis-backed long memory                     |
| 5     | completed | 5512c7e  | flow_executor_node + [FLOW:nome] tag                          |
| 6     | completed | c82f96e  | tenant_resolver_node + Supabase adapter                       |
| 7     | completed | 881c11e  | EvolutionClient as @tool LangGraph                            |
| 8     | completed | a29a18f  | send_node inside graph                                        |
| 9     | completed | 5f044db  | main.py slim + astream_events streaming + checkpointer lifespan |
| 10    | completed | d3fcdd9  | pytest end-to-end suite (21 tests, all passing)               |

## Progress v2 (Production Hardening)

| Phase | Status    | Commit   | What                                                          |
|-------|-----------|----------|---------------------------------------------------------------|
| 11    | completed | 7369555  | summarize_node emite RemoveMessage — trim real do checkpointer |
| 12    | completed | 21626f1  | QStash + `/api/trigger-followup` + KillSwitch redis last_from |
| 13    | completed | d66bd4c  | retry exponential Evolution + slowapi rate limit              |
| 14    | completed | 0c400d6  | Lock Redis por phone (serializa mensagens concorrentes)       |
| 15    | completed | b0b0cbf  | sync_catalog.py + supabase migration products+instance_projects |
| 16    | completed | beb059f  | ToolNode + bind_tools opt-in (ENABLE_TOOL_CALLS=1)            |
| 17    | completed | 56f6ea1  | Dockerfile + railway.toml + nixpacks.toml + Procfile          |
| 18    | completed | c8f3e42  | checkpointer tests (InMemory + skip-if-no-postgres live)      |
| 19    | completed | 2d3bb8d  | scripts/smoke.py (5/5 PASS contra servidor local real)        |
| 20    | completed | (pending)| DEPLOY.md — checklist Railway/Supabase/QStash + troubleshooting |

## Tests

- `tests/test_tools.py` — 12 unit (tags/chunking/flow detection)
- `tests/test_graph.py` — 9 e2e (todos os paths do grafo)
- `tests/test_retry.py` — 4 retry (5xx/429/4xx/timeout)
- `tests/test_checkpointer.py` — 3 + 1 skip (Postgres live se POSTGRES_URL set)
- **Total: 28 passed, 1 skipped, 0 failed**

## Smoke E2E

```
[ok] /health 200
[ok] /webhook 401 sem auth
[ok] /webhook 200 connection.update
[ok] /webhook 200 fromMe skip
[ok] /api/trigger-followup 400 sem instance
[PASS]: 0 failure(s)
```

## Pendente (requer creds usuário)

- Deploy real Railway (`railway up`)
- Supabase migration (`supabase db push` ou cole no SQL editor)
- Configurar webhook na Evolution API
- Smoke E2E em produção: `python -m scripts.smoke --url https://...`

Ver `DEPLOY.md` pra checklist completo.

## Blockers/Concerns

None.
