# Follow-up Intelligent

Sistema de follow-up automático com LLM strategist + scheduler QStash. Pacote portátil.

## O que faz

1. **Classifica temperatura do lead** após cada turn: HOT / WARM / COLD / STOP / SCHEDULED
2. **Decide cadência ótima** baseado em sinais comportamentais (Cialdini)
3. **Extrai horários** mencionados pelo lead ("amanhã 17h", "em 3 min", "fim de semana")
4. **Agenda follow-up** via Upstash QStash (serverless cron)
5. **KillSwitch** automático se lead responde antes do disparo
6. **Hard cap** após N tentativas sem resposta → marca como LOST

## Arquivos

| Arquivo | Função |
|---|---|
| `strategist.py` | LLM classifier + decision validator + fallback |
| `temporal.py` | Regex PT-BR pra horários relativos/absolutos |
| `qstash_client.py` | Wrapper Upstash QStash REST API |
| `example_integration.py` | FastAPI endpoints exemplo |
| `__init__.py` | Public API exports |

## Instalação

```bash
pip install -r requirements.txt
```

Copia a pasta inteira pro seu projeto:

```
seu-projeto/
└── followup_intelligent/
    ├── __init__.py
    ├── strategist.py
    ├── temporal.py
    └── qstash_client.py
```

## Env vars

| Var | Default | Função |
|---|---|---|
| `FOLLOWUP_STRATEGIST_MODEL` | `openai/gpt-4o-mini` | Modelo LLM (OpenAI/OpenRouter) |
| `FOLLOWUP_MAX_ATTEMPTS` | `10` | Cap de tentativas antes de killswitch |
| `FOLLOWUP_STRATEGIST_TIMEOUT` | `15` | Timeout LLM call (segundos) |
| `OPENROUTER_API_KEY` ou `OPENAI_API_KEY` | — | API key pro LLM |
| `AI_BASE_URL` | `https://openrouter.ai/api/v1` | Base URL do provider |
| `QSTASH_TOKEN` | — | Token Upstash QStash |
| `QSTASH_URL` | `https://qstash.upstash.io` | Base URL QStash |
| `PUBLIC_BASE_URL` | — | URL pública do seu app (ex `https://app.up.railway.app`) |
| `WEBHOOK_SECRET` | — | Auth secret pro callback do QStash (forward via header) |

## Uso básico

```python
from followup_intelligent import classify_lead, QStashClient
from langchain_core.messages import HumanMessage, AIMessage

# 1. Histórico da conversa
messages = [
    HumanMessage(content="oi quero saber dos planos"),
    AIMessage(content="claro, tem dois principais: A e B"),
    HumanMessage(content="vou pensar e te falo"),
]

# 2. Classifica
decision = await classify_lead(
    messages=messages,
    last_user_message="vou pensar e te falo",
    attempts_made=0,
)

# decision = {
#   "temperatura": "WARM",
#   "razao": "lead hesitante, disse 'vou pensar'",
#   "agendar_minutos": 120,
#   "abordagem": "valor",
#   "killswitch_permanent": False,
#   "horario_explicito": null,
# }

# 3. Agenda
qstash = QStashClient()
if decision["agendar_minutos"] > 0 and not decision["killswitch_permanent"]:
    result = await qstash.schedule_followup(
        delay_minutes=decision["agendar_minutos"],
        payload={"lead_id": "abc123", "abordagem": decision["abordagem"]},
    )
```

## Integração completa (FastAPI)

Ver `example_integration.py`. Fluxo:

```
Lead manda msg → /webhook/lead-message
                    ↓
              bot responde
                    ↓
           classify_lead()
                    ↓
      qstash.schedule_followup(delay_min)
                    ↓
      (N min depois, QStash dispara)
                    ↓
       POST /api/trigger-followup
                    ↓
   KillSwitch check (lead respondeu?)
                    ↓
       envia msg reengajamento
```

## KillSwitch (essencial)

Quando lead responde antes do disparo agendado, marque last_message_from=lead em storage (Redis recomendado). O endpoint `/api/trigger-followup` checa isso e cancela o envio. Sem isso, bot manda msg DEPOIS do lead voltar → péssima UX.

Exemplo (Redis):

```python
# Lead manda msg
await redis.set(f"last_from:{lead_id}", "lead", ex=86400)

# Endpoint follow-up
last = await redis.get(f"last_from:{lead_id}")
if last == "lead":
    return {"skipped": "lead_replied"}
```

## Schemas

### Decision (output classify_lead)

```python
{
    "temperatura": "HOT" | "WARM" | "COLD" | "STOP" | "SCHEDULED",
    "razao": str,                    # < 120 chars
    "horario_explicito": str | None, # ISO 8601 ou null
    "agendar_minutos": int,          # 0 se STOP, 1-10080 senão
    "abordagem": "commitment" | "valor" | "escassez" | "reciprocidade" | "social",
    "killswitch_permanent": bool,
}
```

### Cadências por temperatura

| Temp | Min minutos | Max minutos | Abordagem default |
|---|---|---|---|
| HOT | 15 | 60 | commitment |
| WARM | 60 | 180 | valor |
| COLD | 720 | 1440 | valor |
| STOP | 0 | 0 | — (killswitch) |
| SCHEDULED | 1 | 10080 | — (do horário detectado) |

## Anti-injection

`strategist.py` sanitiza msgs do lead antes de mandar pro LLM:
- Remove markdown fences (` ``` ` → `ʻʻʻ`)
- Remove pseudo-XML (`<system>`, `<admin>`, etc)
- Trunca 2000 chars
- Substitui ═══ pra não confundir separadores do prompt

## Custos estimados

- Strategist usa `gpt-4o-mini` (input $0.15/M, output $0.60/M)
- Cada classificação ~ 500 tokens in + 100 out = $0.0001
- 1k leads/dia, 3 turns cada = 3k calls = $0.30/dia
- QStash free tier: 500 msgs/dia (suficiente pra MVP)

## Limitações

- Português BR apenas (regex temporal). Pra outros idiomas, adapta `_DAY_KEYWORDS` + `_PERIOD_HOURS` + `_RELATIVE_PATTERNS`.
- LLM call síncrono (não streaming) — adiciona ~1-2s no turn.
- QStash mínimo 1 min de delay (não suporta segundos).

## Não inclui (você implementa)

- Storage de tentativas + last_message_from (use Redis/Postgres)
- Envio de mensagens via canal (WhatsApp/Telegram/SMS)
- Geração do TEXTO do follow-up (use sua lógica de LLM)
- UI admin pra configurar cadências por projeto
