# State

**Milestone:** v1 — LangGraph maximalist
**Current phase:** all complete

## Progress

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

## Tests

- `tests/test_tools.py` — 12 unit tests (tag parsing, chunking, flow detection)
- `tests/test_graph.py` — 9 e2e tests (greeting/close/objection/comprou/vision/flow/tenant/persist/react paths)
- **Total: 21 passed, 0 failed**

## Blockers/Concerns

None.
