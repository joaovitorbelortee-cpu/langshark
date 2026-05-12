# Features portáteis

Pacotes standalone extraídos do bot-vendas pra reutilizar em outros projetos.

Cada pasta = pacote independente com README de integração próprio.

## Pacotes

### 📅 [followup_intelligent/](./followup_intelligent/)

Sistema de follow-up automático com LLM strategist + scheduler QStash.

- Classifica temperatura do lead (HOT/WARM/COLD/STOP/SCHEDULED)
- Decide cadência ótima por psicologia (Cialdini)
- Extrai horários PT-BR ("amanhã 17h", "em 3 min")
- Agenda via Upstash QStash
- KillSwitch se lead responde
- Hard cap configurable

Deps: `langchain-openai`, `httpx`. Stack-agnostic.

### 🎬 [flows/](./flows/)

Sequências pré-cadastradas (text/image/video/audio/document) disparáveis por tag `[FLOW: nome]`.

- Catálogo persistido em Supabase (multi-tenant, cache 60s)
- Painel admin FastAPI CRUD incluído
- MessageSender Protocol → canal-agnostic (WhatsApp/Telegram/Discord/etc)
- Fallback safe: texto do LLM se flow falhar

Deps: `httpx`, `fastapi` (opcional pro painel). Stack-agnostic.

## Como combinar

Os 2 pacotes são independentes mas se complementam: bot usa **flows** pra disparar mídia rica quando lead pede demo, e usa **followup-intelligent** pra agendar reengajamento quando lead some. Exemplo combinado:

```python
from followup_intelligent import classify_lead, QStashClient
from flows import parse_flow_tag, execute_flow, flows_prompt_block

# 1. Inject flows no system prompt
system = base_prompt + "\n\n" + flows_prompt_block(project_id)

# 2. LLM gera reply
raw_reply = await llm.ainvoke(system + history)

# 3. Dispatch flow se tem tag, senão texto
flow_name, text = parse_flow_tag(raw_reply)
if flow_name:
    await execute_flow(project_id, flow_name, lead_phone, sender)
else:
    await sender.send_text(lead_phone, text)

# 4. Classifica follow-up
decision = await classify_lead(messages, last_user_msg, attempts)
if decision["agendar_minutos"] > 0:
    await qstash.schedule_followup(
        delay_minutes=decision["agendar_minutos"],
        payload={"lead_id": lead_phone, "abordagem": decision["abordagem"]},
    )
```

## Origem

Extraído de [`bot-vendas`](https://github.com/joaovitorbelortee-cpu/langshark) — bot WhatsApp de vendas Game Pass. Código rodando em produção desde maio 2026.

## Licença

Use à vontade. Sem garantias.
