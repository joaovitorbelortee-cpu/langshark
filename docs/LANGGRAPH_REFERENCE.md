# LangGraph reference — padrões usados no bot-vendas

Mapa entre **docs oficiais LangGraph** e **decisões do projeto**. Atualizar
sempre que migrar versão LangGraph ou mudar arquitetura.

Fontes oficiais (lidas em sessão de migração):
- StateGraph signatures: `libs/langgraph/langgraph/graph/state.py` (main branch)
- BaseStore signatures: `libs/checkpoint/langgraph/store/base/__init__.py`
- Runtime: `libs/langgraph/langgraph/runtime.py`
- Persistence concept: `docs.langchain.com/oss/python/langgraph/persistence`
- Streaming: `docs.langchain.com/oss/python/langgraph/streaming`

---

## 1. StateGraph — assinaturas que o projeto usa

```python
StateGraph(
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
)
```

**No projeto** (`agent/state.py`):
- `state_schema = SalesState` (TypedDict)
- Não usa `context_schema` (Runtime[Context] não injetado)

### `add_node`

```python
graph.add_node(
    node: str | StateNode,
    action: StateNode | None = None,
    *,
    defer: bool = False,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
    error_handler: StateNode | None = None,
)
```

**No projeto** (`agent/graph.py`): usa só `g.add_node(name, fn)`. Sem
retry/cache/timeout per-node — fica em TODO se quisermos hardening.

### `add_conditional_edges`

```python
graph.add_conditional_edges(
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
)
```

**No projeto**: usado pra rotear após `detect_intent`, `summarize`, `reply_node`,
`supervisor`. Returns string key, mapeado pra node names via dict.

### `compile`

```python
graph.compile(
    checkpointer: Checkpointer = None,
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
)
```

**No projeto** (`agent/graph.py:build_graph`):
- ✓ `checkpointer` (AsyncRedisSaver/AsyncPostgresSaver/InMemorySaver)
- ✓ `store` (AsyncRedisStore/InMemoryStore) — migração 2026-05
- ✗ `interrupt_before/after` — não usado (sem human-in-the-loop)
- ✗ `cache` — não usado

---

## 2. BaseStore — API correta

```python
# Read
async def aget(namespace: tuple[str, ...], key: str) -> Item | None
async def asearch(namespace_prefix: tuple[str, ...], *, query: str | None,
                  filter: dict | None, limit: int = 10, offset: int = 0) -> list[SearchItem]

# Write
async def aput(namespace: tuple[str, ...], key: str, value: dict,
               index: bool | list[str] | None = None, *, ttl: float | None)
async def adelete(namespace: tuple[str, ...], key: str)

# Discover
async def alist_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100)
```

**Item retornado por aget**: `Item(namespace, key, value, created_at, updated_at)`.
`SearchItem` (de asearch) adiciona `score`.

**No projeto** (`agent/store.py` + `agent/nodes.py:lead_memory_node`):
- Namespace canônico: `(project_id, phone)` — uma "pasta" por lead.
- Key: `"facts"` (singleton por lead).
- Value: dict `lead_facts` (plataforma, plano_interesse, estagio, ...).

**Padrão correto** (lido docs):
```python
# Nó signature recomendada
async def my_node(state: State, runtime: Runtime[Context]) -> dict:
    if runtime.store:
        item = await runtime.store.aget((user_id, "memories"), "key")
        # item.value pra acessar dict
```

**Decisão do projeto**: usar `agent.store.get_shared_store()` (módulo
singleton) em vez de `runtime: Runtime[Context]`. Motivos:
1. Runtime API teve breaking changes entre 0.2/0.3. Singleton é estável.
2. Nós são funções puras sem dataclass de Context.
3. Singleton facilita testes (`set_shared_store(InMemoryStore())`).

Trade-off: menos idiomático mas igual robusto.

---

## 3. Checkpointer — hierarquia + thread

**thread_id format** (do projeto, `agent/checkpointer.py:thread_id_for`):
```
{project_id}:{instance_name}:{phone}
```

Garante que mesma conversa (lead único) tem mesmo thread_id mesmo
restartando bot. Checkpoint serializa via JsonPlusSerializer.

**Hierarquia**:
1. `REDIS_URL` → `AsyncRedisSaver` (preferido)
2. `POSTGRES_URL` → `AsyncPostgresSaver` (fallback)
3. Else → `InMemorySaver` (dev only, perde state em restart)

**APIs usadas no projeto**:
- `graph.ainvoke(state, config={"configurable": {"thread_id": ...}})` ✓
- `graph.astream(...)` em `main.py:_run_graph_streaming` (stream mode `values`
  pra trace por nó). Poderia migrar pra v2 com `stream_mode=["updates"]`.

**APIs disponíveis NÃO usadas (futuro)**:
- `graph.aget_state(config)` — inspecionar checkpoint atual
- `graph.aupdate_state(config, values)` — modificar state mid-run
- `graph.aget_state_history(config)` — time-travel debugging

---

## 4. State schema — reducers

```python
class SalesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # ✓ reducer correto
    # outros campos: NotRequired[...] sem reducer (overwrite default)
```

**`add_messages`** (de `langgraph.graph.message`): merge inteligente que:
- Append novas mensagens
- Dedup por message ID
- Suporta `RemoveMessage(id=...)` pra apagar
- Aceita lista de dicts → converte pra BaseMessage

**No projeto**: usado em `messages`. Outros campos overwrite — correto pra
state derivativo (intent recalculado todo turno, reply é resultado do turno).

---

## 5. Streaming — modos disponíveis

Docs oficiais listam:
| Mode | Use case |
|------|----------|
| `values` | Snapshot completo após cada step |
| `updates` | Apenas keys mudados por nó |
| `messages` | Tokens LLM streaming |
| `custom` | Dados emitidos via `get_stream_writer()` |
| `checkpoints` | Eventos de checkpoint |
| `tasks` | Início/fim de tasks |
| `debug` | Combinação checkpoint+tasks |

**No projeto** (`main.py:_run_graph_streaming`): usa stream pra logar
"per-node trace". Modo atual = legacy (não passa version="v2"). Funcional
mas poderia migrar pro v2.

**Tokens streaming não usado** — bot envia chunks por WhatsApp via
`chunk_for_whatsapp`, então streaming tokens não traria valor end-user.

---

## 6. Custom data emission

```python
from langgraph.config import get_stream_writer

async def my_node(state):
    writer = get_stream_writer()
    writer({"status": "carregando catalog..."})
    ...
```

**No projeto**: não usado. Poderia emitir progresso no painel admin (futuro).

---

## 7. Tools (ToolNode) — versão atual

**No projeto** (`agent/graph.py`):
```python
g.add_node("tools", ToolNode(EVOLUTION_TOOLS))
```

ToolNode auto-executa tool_calls da última AIMessage e adiciona
ToolMessage no state. Padrão correto.

`_route_after_reply` em `agent/graph.py` checa `_last_ai_has_tool_calls` e
rota pra `tools_path` quando AIMessage tem tool_calls + flag
`ENABLE_TOOL_CALLS` está on.

---

## 8. Versões alvo

- `langgraph>=0.2.50` (current)
- `langgraph-checkpoint-postgres>=2.0.0`
- `langgraph-checkpoint-redis>=0.0.6` ← NEW

Subir pra 0.3.x se quiser:
- `Runtime[Context]` injection mais limpo nos nós
- API estabilizada de Store com semantic search via embeddings
- Melhorias de performance no stream

**Não migrar agora** — 0.2.50 está estável e tem tudo que precisamos.

---

## 9. TODO — Hardening sugerido pelos docs

| Item | Por quê | Onde aplicar |
|------|--------|--------------|
| `retry_policy=RetryPolicy(max_attempts=2)` nos nós LLM | Recuperação automática de timeout/rate-limit | `respond/close_sale/follow_up/objection/greeting/supervisor/lead_memory/strategist` |
| `timeout` per-node | Bound max tempo por nó (evita webhook travado) | Mesmos nós LLM |
| `interrupt_before=["send"]` | Em projetos com aprovação manual | Não necessário aqui |
| Vector index no AsyncRedisStore | Semantic recall de turnos passados | Próxima iteração |
| Migrar `astream` pra v2 + `stream_mode=["updates"]` | Logs estruturados melhor | `_run_graph_streaming` |

---

## 10. Anti-padrões evitados

- ✗ Não criar `_make_llm` global compartilhado entre threads (LangChain
  recomenda 1 instância por chamada — usamos via `_make_llm()` factory).
- ✗ Não mutar `state` diretamente (sempre retornar patch dict).
- ✗ Não fazer side-effects fora dos nós (DB writes só dentro de nodes).
- ✗ Não criar threads concorrentes pro mesmo thread_id (lock per phone protege).
