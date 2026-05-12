# Flows — Sequências pré-cadastradas

Sistema de fluxos disparáveis por tag `[FLOW: nome]` na resposta do LLM. Canal-agnostic.

## O que faz

1. **Catálogo de flows** persistido em Supabase (multi-tenant, cache 60s)
2. **Cada flow = lista de steps**: text / image / video / audio / document
3. **LLM emite `[FLOW: nome]`** → bot dispara sequência via Evolution/Telegram/etc
4. **Painel admin** (FastAPI router incluso) pra CRUD flows
5. **Fallback safe**: se todos steps falham, bot envia texto do LLM como backup

## Arquivos

| Arquivo | Função |
|---|---|
| `flows.py` | Supabase fetch + registry in-memory + parse_flow_tag |
| `flow_executor.py` | `execute_flow()` + `MessageSender` Protocol |
| `panel_endpoints.py` | FastAPI CRUD `/api/admin/flows/*` |
| `migrations/0001_flows_schema.sql` | Tabela `public.flows` + RLS |
| `migrations/0002_flow_media_bucket.sql` | Storage bucket `flow-media` (uploads) |
| `example_integration.py` | Exemplo com Evolution API (WhatsApp) |
| `__init__.py` | Public API |

## Instalação

```bash
pip install -r requirements.txt
```

Copia a pasta:

```
seu-projeto/
└── flows/
    ├── __init__.py
    ├── flows.py
    ├── flow_executor.py
    └── panel_endpoints.py
```

Migrate Supabase:

```bash
psql $POSTGRES_URL -f migrations/0001_flows_schema.sql
psql $POSTGRES_URL -f migrations/0002_flow_media_bucket.sql
```

## Env vars

| Var | Default | Função |
|---|---|---|
| `SUPABASE_URL` | — | URL Supabase REST API |
| `SUPABASE_SERVICE_KEY` | — | service_role key (bypass RLS) |
| `SUPABASE_ALLOW_ANON` | — | =1 permite anon fallback (NÃO usar em prod) |

Sem Supabase configurado, usa apenas `FLOW_REGISTRY` in-memory (chama `register_flow()` manualmente).

## Schema da tabela `public.flows`

```sql
id          uuid PRIMARY KEY
project_id  text NOT NULL          -- multi-tenant key
name        text NOT NULL          -- ex: "video_pitch"
description text DEFAULT ''        -- "quando usar" — vai pro prompt LLM
steps       jsonb DEFAULT '[]'     -- lista de steps (schema abaixo)
enabled     boolean DEFAULT true   -- toggle on/off sem deletar
created_at  timestamptz
updated_at  timestamptz
UNIQUE(project_id, name)
```

## Schema do step (JSONB)

```json
// Texto
{"type": "text", "content": "Olá, tudo bem?"}

// Imagem
{"type": "image", "url": "https://cdn.../img.jpg", "caption": "Foto do produto"}

// Vídeo
{"type": "video", "url": "https://cdn.../pitch.mp4", "caption": "Demo de 30s"}

// Áudio
{"type": "audio", "url": "https://cdn.../msg.ogg"}

// Documento
{"type": "document", "url": "https://cdn.../catalogo.pdf", "fileName": "Catálogo.pdf", "caption": "Tabela completa"}
```

## Uso básico (1 LLM call + dispatch)

```python
from flows import parse_flow_tag, execute_flow, get_flow, flows_prompt_block

# 1. Injeta lista de flows disponíveis no system prompt
system_prompt = f"""
Você é um vendedor por WhatsApp.

{flows_prompt_block("padrao")}
"""

# 2. LLM gera reply (pode ter tag [FLOW: nome])
raw_reply = await your_llm.ainvoke(system_prompt + user_msg)
# Ex: "Vou te mostrar agora! [FLOW: video_pitch]"

# 3. Parse + dispatch
flow_name, cleaned_text = parse_flow_tag(raw_reply)

if flow_name and get_flow("padrao", flow_name):
    result = await execute_flow(
        project_id="padrao",
        flow_name=flow_name,
        to=lead_phone,
        sender=your_whatsapp_client,  # implementa MessageSender Protocol
    )
    if not result["dispatched"]:
        # Fallback: nenhum step enviou — manda texto do LLM
        await your_whatsapp_client.send_text(lead_phone, cleaned_text)
else:
    await your_whatsapp_client.send_text(lead_phone, cleaned_text)
```

## MessageSender Protocol

Implemente 3 métodos pro seu canal:

```python
class MyChannelSender:
    async def send_text(self, to: str, text: str) -> Any: ...

    async def send_typing(self, to: str, duration_ms: int = 1200) -> Any:
        """Opcional — pode ser no-op se canal não suporta."""

    async def send_media(
        self,
        to: str,
        kind: str,         # "image" | "video" | "audio" | "document"
        url: str,
        caption: str | None = None,
        file_name: str | None = None,
    ) -> Any: ...
```

Exemplo Evolution API (WhatsApp) em `example_integration.py`.

## Painel Admin (CRUD)

```python
from fastapi import FastAPI
from flows.panel_endpoints import router as flows_router

app = FastAPI()
app.include_router(flows_router, prefix="/api/admin")

# Override require_admin pelo SEU dependency de auth:
from flows import panel_endpoints
panel_endpoints.require_admin = your_real_auth_dependency
```

Endpoints:

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/admin/flows?project_id=padrao` | Lista flows do projeto |
| POST | `/api/admin/flows` | Cria flow (`{project_id, name, description, steps, enabled}`) |
| PATCH | `/api/admin/flows/{id}` | Atualiza campos do flow |
| DELETE | `/api/admin/flows/{id}` | Deleta flow |

Cache é invalidado automaticamente em cada mutation.

## System prompt — diga ao LLM como usar flows

`flows_prompt_block(project_id)` gera bloco automático tipo:

```
<fluxos_cadastrados>
Você pode acionar um fluxo pré-gravado emitindo a tag [FLOW: nome] no FINAL da resposta.
O sistema enviará a sequência cadastrada e ignorará o texto da resposta atual.
REGRA: só dispare flow quando a situação descrita em 'quando usar' bater CLARAMENTE
com a mensagem do lead. Em dúvida, NÃO dispare — responda normal.
Fluxos disponíveis:
- video_pitch — quando usar: Cliente pediu pra ver demonstração ou material visual
- catalogo_pdf — quando usar: Cliente pediu lista completa de produtos
</fluxos_cadastrados>
```

Concatena no seu system prompt antes de mandar pro LLM.

## Cache

- TTL 60s por project_id
- LRU eviction quando atinge 100 projetos
- `invalidate_flows_cache(project_id)` força refresh (chamado pelos endpoints admin)

## Fallback safety

Se TODOS os steps de um flow falham (Evolution down, URL inválida, etc), `execute_flow()` retorna `dispatched=False`. Caller deve enviar texto do LLM como fallback pra lead não ficar em silêncio.

```python
result = await execute_flow(...)
if not result["dispatched"]:
    await sender.send_text(lead_phone, cleaned_text or "Te respondo já já")
```

## Multi-tenant

Cada `project_id` tem seus próprios flows. Pra setup single-tenant, use `project_id="padrao"` em tudo.

## Não inclui (você implementa)

- LLM call (use OpenAI/Anthropic/etc)
- Auth do painel admin (substitua `require_admin` placeholder)
- Upload UI (use bucket `flow-media` + Supabase signed URLs)
- Validação de URLs nos steps (recomendado adicionar antes de salvar)
