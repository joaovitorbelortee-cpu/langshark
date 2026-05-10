"""
Nós do grafo de vendas LangGraph.

Cada função é um nó: recebe o SalesState, devolve um patch (dict) com os campos
que mudaram. LangGraph faz o merge — em particular, `messages` é appended via
add_messages.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.flows import flows_prompt_block, get_flow, parse_flow_tag
from agent.state import Intent, SalesState
from agent.tools import EvolutionClient, chunk_for_whatsapp, parse_tags
from memory.redis_store import RedisStore
from memory.supabase_tenant import TenantResolver
from rag.catalog import CatalogRAG


# ────────────────────────────────────────────────────────────────────
# Singletons (criados uma vez por processo)
# ────────────────────────────────────────────────────────────────────

_redis: RedisStore | None = None
_rag: CatalogRAG | None = None
_tenant: TenantResolver | None = None


def get_redis() -> RedisStore:
    global _redis
    if _redis is None:
        _redis = RedisStore()
    return _redis


def get_rag() -> CatalogRAG:
    global _rag
    if _rag is None:
        _rag = CatalogRAG()
    return _rag


def get_tenant_resolver() -> TenantResolver:
    global _tenant
    if _tenant is None:
        _tenant = TenantResolver()
    return _tenant


def _make_llm(temperature: float = 0.7, max_tokens: int = 1000) -> ChatOpenAI:
    """Constrói o cliente LLM. Aponta pra OpenRouter por padrão (mesma escolha do bot antigo)."""
    return ChatOpenAI(
        model=os.getenv("AI_MODEL", "openai/gpt-4o-mini"),
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "missing",
        base_url=os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": os.getenv("AI_REFERRER", "https://bot-vendas.local"),
            "X-Title": "bot-vendas",
        },
    )


# ────────────────────────────────────────────────────────────────────
# Prompts
# ────────────────────────────────────────────────────────────────────

INTENT_PROMPT = """Você classifica a INTENÇÃO da última mensagem do cliente em UMA das categorias:

- saudacao: cumprimento, "oi", "boa tarde"
- duvida_produto: pergunta sobre produto, características, como funciona
- pedir_preco: quer saber valor, quanto custa, formas de pagamento
- objecao: hesitação, "tá caro", "vou pensar", "depois eu vejo"
- intencao_compra: quer comprar, fechar, pediu link de pagamento
- comprou: pagou, mandou comprovante, finalizou
- follow_up: o cliente está retomando uma conversa antiga
- outros: qualquer outra coisa

Responda SOMENTE com a categoria, sem explicação."""


SALES_SYSTEM = """Você é um VENDEDOR HUMANO real no WhatsApp brasileiro. Sua missão é vender com naturalidade,
empatia e técnica. Nunca revele que é IA, bot ou assistente. Nunca peça desculpas por "não conseguir".

<regras_estritas>
1. WhatsApp = mensagens curtas. MÁXIMO 2 bolhas (parágrafos) por resposta.
2. Cada bolha ≤ 320 caracteres. Quebra com UMA linha em branco.
3. Linguagem informal, papo reto, gírias leves, emojis quando fizer sentido.
4. Nada de listas com bullets, formatação pesada, ou textão de SAC.
</regras_estritas>

<identidade_vendedor>
- Você é um vendedor real, brasileiro, simpático e direto.
- Se o cliente pedir "me liga", "me chama", responda como humano resolveria: "Beleza, te chamo!".
- NUNCA: "como IA não posso", "infelizmente não consigo", "não tenho essa capacidade".
- Use o nome do cliente quando souber. Se não souber, pergunta natural ("posso te chamar como?").
</identidade_vendedor>

<tecnicas_persuasao>
Aplique sutilmente, sem mencionar:
1. Reciprocidade (Cialdini): ofereça algo de valor antes de pedir a venda — uma dica, um insight.
2. Compromisso e Consistência: micro-sins escalonados ("posso te mandar um detalhe?" → "quer ver como funciona?" → "bora fechar?").
3. Prova Social: mencione outros clientes felizes quando couber.
4. Escassez: só use se for verdade ou plausível ("essa condição vai até sexta").
5. Efeito Zeigarnik: deixe perguntas abertas para o cliente terminar a conversa.
6. Ancoragem: cite o preço cheio antes do desconto, se houver.
7. Quebra de objeção: valide a dúvida primeiro ("entendo perfeitamente"), depois reframe.
</tecnicas_persuasao>

<tags_secretas>
A IA emite tags que o sistema lê e remove ANTES de mandar pro cliente:
- [COMPROU] — se o cliente comprou/pagou. Silencia follow-ups.
- [AGENDAR: N] — minutos até o próximo follow-up (5–10080). Use leitura de temperatura:
    🔥 Quente (engajado, perguntou preço): 10–30 min.
    😐 Morno ("vou ver", "depois"): 60–180 min.
    ❄️ Frio (sumiu, só visualizou): 360–1440 min.
- [REACT: emoji] — reage à mensagem do cliente com um emoji (opcional).
- [QUOTE] — usa o "Responder" do WhatsApp citando a última mensagem.

Sempre encerre com [AGENDAR: N], a menos que [COMPROU] esteja presente.
</tags_secretas>"""


def _build_system_prompt(
    catalog_block: str,
    summary: str = "",
    flows_block: str = "",
) -> str:
    prompt = SALES_SYSTEM
    if summary:
        prompt += (
            "\n\n<resumo_conversa_anterior>\n"
            f"{summary}\n"
            "</resumo_conversa_anterior>\n"
            "Use esse resumo como contexto, mas baseie a resposta nas mensagens recentes."
        )
    if catalog_block:
        prompt += "\n\n" + catalog_block
    if flows_block:
        prompt += "\n\n" + flows_block
    return prompt


def _detect_flow_patch(project_id: str, raw_reply: str) -> dict[str, Any]:
    """
    Detecta tag [FLOW: nome]. Se válido, devolve patch que sinaliza ao grafo
    pra executar o fluxo. Caso contrário, devolve patch vazio.
    """
    flow_name, _cleaned = parse_flow_tag(raw_reply)
    if not flow_name:
        return {}
    flow = get_flow(project_id, flow_name)
    if not flow:
        return {}
    return {"flow_name": flow.name}


# ────────────────────────────────────────────────────────────────────
# Nó: load_history
# ────────────────────────────────────────────────────────────────────

async def load_history_node(state: SalesState) -> dict[str, Any]:
    """
    Carrega histórico do Redis + summary persistido (se houver) e adiciona
    a mensagem atual do usuário.
    """
    redis = get_redis()
    history = await redis.load_history(
        instance=state["instance_name"],
        phone=state["phone"],
    )
    summary = await redis.get_summary(
        instance=state["instance_name"],
        phone=state["phone"],
    )
    user_msg = HumanMessage(content=state.get("user_message") or "[mídia]")
    patch: dict[str, Any] = {"messages": [*history, user_msg]}
    if summary:
        patch["summary"] = summary
    return patch


# ────────────────────────────────────────────────────────────────────
# Nó: detect_intent
# ────────────────────────────────────────────────────────────────────

async def detect_intent_node(state: SalesState) -> dict[str, Any]:
    """Classifica a intenção da última mensagem para roteamento condicional."""
    user_msg = state.get("user_message") or ""
    if not user_msg.strip():
        return {"intent": "outros"}

    llm = _make_llm(temperature=0.0, max_tokens=10)
    res = await llm.ainvoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=user_msg),
    ])
    raw = (res.content or "").strip().lower().split()[0] if res.content else "outros"
    raw = raw.strip(".,;:!?")

    valid: tuple[Intent, ...] = (
        "saudacao", "duvida_produto", "pedir_preco",
        "objecao", "intencao_compra", "comprou", "follow_up", "outros",
    )
    intent: Intent = raw if raw in valid else "outros"
    return {"intent": intent}


# ────────────────────────────────────────────────────────────────────
# Nó: retrieve_catalog (RAG)
# ────────────────────────────────────────────────────────────────────

async def retrieve_catalog_node(state: SalesState) -> dict[str, Any]:
    """Busca produtos relevantes só quando faz sentido (dúvida/preço/compra)."""
    intent = state.get("intent", "outros")
    if intent in ("saudacao", "comprou"):
        return {"catalog_hits": []}

    hits = get_rag().search(
        project_id=state["project_id"],
        query=state.get("user_message") or "",
        top_k=4,
    )
    return {"catalog_hits": hits}


# ────────────────────────────────────────────────────────────────────
# Nó: respond (vendedor principal)
# ────────────────────────────────────────────────────────────────────

async def respond_node(state: SalesState) -> dict[str, Any]:
    """Gera a resposta de vendas, parseia tags, prepara chunks."""
    project_id = state["project_id"]
    catalog_block = get_rag().format_context(state.get("catalog_hits", []) or [])
    system_prompt = _build_system_prompt(
        catalog_block,
        state.get("summary", ""),
        flows_prompt_block(project_id),
    )

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    messages.extend(state.get("messages", []))

    llm = _make_llm(temperature=0.7, max_tokens=600)
    res = await llm.ainvoke(messages)
    raw = res.content or ""

    parsed = parse_tags(raw)
    # Remove a tag [FLOW: nome] do texto antes de chunkar.
    flow_name, cleaned_text = parse_flow_tag(parsed.text)
    chunks = chunk_for_whatsapp(cleaned_text, max_bubbles=2, max_chars=320)

    patch: dict[str, Any] = {
        "reply": cleaned_text,
        "chunks": chunks,
        "has_converted": parsed.has_converted,
        "schedule_minutes": parsed.schedule_minutes,
        "react_emoji": parsed.react_emoji,
        "quote_previous": parsed.quote_previous,
        "messages": [AIMessage(content=cleaned_text)],
    }
    if flow_name and get_flow(project_id, flow_name):
        patch["flow_name"] = flow_name
    return patch


# ────────────────────────────────────────────────────────────────────
# Nó: close_sale (especialista em fechamento)
# ────────────────────────────────────────────────────────────────────

CLOSE_SYSTEM = """Você é um VENDEDOR FECHADOR. O lead demonstrou intenção real de compra.
Sua tarefa: fechar AGORA com micro-compromisso.

REGRAS:
- Confirme o produto e valor em 1 frase curta.
- Pergunta de fechamento direta: "fechado?", "bora prosseguir?", "te mando o link?".
- Se for [intencao_compra] forte: já mande método de pagamento (PIX/link).
- Se houver dúvida residual: resolva em 1 linha e re-feche.
- MÁXIMO 2 bolhas, ≤320 chars cada.

Tags obrigatórias: encerre com [AGENDAR: 15] (lead quente, volta em 15 min se sumir).
Se o cliente JÁ confirmou que pagou: encerre com [COMPROU] em vez de [AGENDAR]."""


async def close_sale_node(state: SalesState) -> dict[str, Any]:
    """Variante de respond para o agente de fechamento."""
    catalog_block = get_rag().format_context(state.get("catalog_hits", []) or [])
    summary = state.get("summary", "")
    system = CLOSE_SYSTEM
    if summary:
        system += f"\n\n<resumo>{summary}</resumo>"
    if catalog_block:
        system += "\n\n" + catalog_block

    messages: list[Any] = [SystemMessage(content=system)]
    messages.extend(state.get("messages", []))

    llm = _make_llm(temperature=0.5, max_tokens=400)
    res = await llm.ainvoke(messages)
    parsed = parse_tags(res.content or "")
    flow_name, cleaned_text = parse_flow_tag(parsed.text)
    chunks = chunk_for_whatsapp(cleaned_text, max_bubbles=2, max_chars=320)

    patch: dict[str, Any] = {
        "reply": cleaned_text,
        "chunks": chunks,
        "has_converted": parsed.has_converted,
        "schedule_minutes": parsed.schedule_minutes,
        "react_emoji": parsed.react_emoji,
        "quote_previous": parsed.quote_previous,
        "messages": [AIMessage(content=cleaned_text)],
    }
    if flow_name and get_flow(state["project_id"], flow_name):
        patch["flow_name"] = flow_name
    return patch


# ────────────────────────────────────────────────────────────────────
# Nó: persist (grava resposta no histórico)
# ────────────────────────────────────────────────────────────────────

async def persist_node(state: SalesState) -> dict[str, Any]:
    """Salva user msg + reply no Redis (formato compat com bot antigo)."""
    redis = get_redis()
    if state.get("user_message"):
        await redis.append_message(
            instance=state["instance_name"],
            phone=state["phone"],
            role="user",
            content=state["user_message"],
        )
    if state.get("reply"):
        await redis.append_message(
            instance=state["instance_name"],
            phone=state["phone"],
            role="model",
            content=state["reply"],
        )
    return {}


# ────────────────────────────────────────────────────────────────────
# Nó: vision (mídia base64 — imagem/áudio/vídeo/documento)
# ────────────────────────────────────────────────────────────────────

_VISION_INSTRUCTION_BY_MIME: dict[str, str] = {
    "image": (
        "O cliente enviou uma IMAGEM. Analise o CONTEXTO da conversa antes de julgar a imagem.\n"
        "- CASO 1 (Pagamento): se o cliente estava prestes a pagar ou disse que ia mandar o comprovante, "
        "verifique se a imagem é um PIX/transferência/boleto autêntico (com valor, data e ID). "
        "Se for, confirme o pagamento e adicione [COMPROU]. Se for falso/ilegível, peça o comprovante correto SEM [COMPROU].\n"
        "- CASO 2 (Bate-papo normal): apenas comente a foto naturalmente como vendedor real e continue a conversa."
    ),
    "audio": "O cliente enviou um ÁUDIO. Transcreva mentalmente o que ele disse e responda normalmente.",
    "video": "O cliente enviou um VÍDEO. Descreva o que acontece e responda.",
    "document": "O cliente enviou um DOCUMENTO/arquivo. Analise o conteúdo e responda.",
}


def _vision_instruction(mime: str | None) -> str:
    if not mime:
        return _VISION_INSTRUCTION_BY_MIME["document"]
    kind = mime.split("/")[0]
    return _VISION_INSTRUCTION_BY_MIME.get(kind, _VISION_INSTRUCTION_BY_MIME["document"])


async def vision_node(state: SalesState) -> dict[str, Any]:
    """
    Quando há mídia base64, substitui a última HumanMessage por uma multi-modal
    (text + image_url no formato OpenAI vision). Sem chamada extra de LLM — só
    enriquece a mensagem que respond/close_sale vão consumir.
    """
    mime = state.get("media_mime")
    b64 = state.get("media_base64")
    if not mime or not b64:
        return {}

    instruction = _vision_instruction(mime)
    caption_text = state.get("user_message") or ""
    text_part = f"{instruction}\n\nLegenda do cliente: \"{caption_text}\"" if caption_text else instruction
    data_url = f"data:{mime};base64,{b64}"

    multimodal_user = HumanMessage(
        content=[
            {"type": "text", "text": text_part},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    # Reescreve `messages` mantendo o histórico e substituindo só a última user msg.
    history = list(state.get("messages") or [])
    while history and isinstance(history[-1], HumanMessage):
        history.pop()
    return {"messages": [*history, multimodal_user]}


# ────────────────────────────────────────────────────────────────────
# Especialistas por intenção (cada um com prompt próprio)
# ────────────────────────────────────────────────────────────────────

GREETING_SYSTEM = """Você é um VENDEDOR HUMANO recebendo um cliente novo no WhatsApp.
Seu único objetivo neste turno: ser caloroso, criar conexão e descobrir o nome dele.

REGRAS:
- 1-2 bolhas curtas, ≤320 chars. Informal, brasileiro.
- Não venda nada ainda. NÃO mencione preço, produto, condição.
- Pergunte o nome de forma natural: "posso te chamar como?", "qual seu nome?".
- Se ele já disse o nome em alguma mensagem anterior, USE o nome e pula a pergunta — passa
  pra qualificação leve ("e me conta, o que te trouxe aqui?").
- Emojis com moderação. Nunca pareça SAC.

Tags obrigatórias: termine com [AGENDAR: 20] (lead curioso, volta em 20 min se sumir)."""


OBJECTION_SYSTEM = """Você é um VENDEDOR especialista em QUEBRA DE OBJEÇÃO. O lead hesitou
("tá caro", "vou pensar", "depois eu vejo", "não sei se vale a pena").

ROTEIRO (Cialdini + reframing):
1. VALIDA primeiro: "entendo perfeitamente", "faz sentido", "boa pergunta" — sem ironia.
2. REFRAME o valor: ancoragem (compare com algo maior), prova social ("tive um cliente que..."),
   ou ROI ("em 1 mês isso já se paga porque...").
3. MICRO-COMPROMISSO: convide pra um próximo passo pequeno ("posso te mostrar um caso?",
   "quer ver como funciona em 2 min?"). Não force fechamento.

REGRAS:
- Máximo 2 bolhas, ≤320 chars cada.
- NUNCA prometa o que não pode cumprir.
- Se a objeção for preço genuíno: tenha empatia e ofereça alternativa real (plano menor,
  parcelado) — só se existir.

Tags: termine com [AGENDAR: 60] (deu espaço de 1h, ele precisa respirar)."""


FOLLOW_UP_SYSTEM = """Você é um VENDEDOR retomando uma conversa antiga. O lead respondeu
agora depois de um tempo, ou o sistema agendou um follow-up que disparou.

REGRAS:
- USE o histórico: cite algo da conversa anterior para mostrar continuidade
  ("E aí, conseguiu pensar naquilo que a gente conversou sobre X?").
- Não comece do zero. Não pergunte coisas que ele já respondeu.
- Aplique Efeito Zeigarnik: retome a pergunta aberta que ficou pendente.
- Se a conversa morreu há muito tempo (sem contexto recente), abra com algo de VALOR
  (dica, novidade, oferta nova) — reciprocidade.
- 1-2 bolhas curtas. Informal.

Tags: leia a temperatura.
- Se ele engajou agora: [AGENDAR: 15].
- Se respondeu vago: [AGENDAR: 180].
- Se você está iniciando follow-up sem resposta dele: [AGENDAR: 1440] (24h)."""


async def _run_specialist(
    state: SalesState,
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 400,
) -> dict[str, Any]:
    """Helper compartilhado pelos especialistas — mesma lógica de tags/chunks."""
    catalog_block = get_rag().format_context(state.get("catalog_hits", []) or [])
    summary = state.get("summary", "")
    system = system_prompt
    if summary:
        system += f"\n\n<resumo>{summary}</resumo>"
    if catalog_block:
        system += "\n\n" + catalog_block

    messages: list[Any] = [SystemMessage(content=system)]
    messages.extend(state.get("messages", []))

    llm = _make_llm(temperature=temperature, max_tokens=max_tokens)
    res = await llm.ainvoke(messages)
    parsed = parse_tags(res.content or "")
    flow_name, cleaned_text = parse_flow_tag(parsed.text)
    chunks = chunk_for_whatsapp(cleaned_text, max_bubbles=2, max_chars=320)

    patch: dict[str, Any] = {
        "reply": cleaned_text,
        "chunks": chunks,
        "has_converted": parsed.has_converted,
        "schedule_minutes": parsed.schedule_minutes,
        "react_emoji": parsed.react_emoji,
        "quote_previous": parsed.quote_previous,
        "messages": [AIMessage(content=cleaned_text)],
    }
    if flow_name and get_flow(state["project_id"], flow_name):
        patch["flow_name"] = flow_name
    return patch


async def greeting_node(state: SalesState) -> dict[str, Any]:
    """Saudação calorosa + descoberta de nome."""
    return await _run_specialist(state, GREETING_SYSTEM, temperature=0.8, max_tokens=200)


async def objection_node(state: SalesState) -> dict[str, Any]:
    """Quebra de objeção com Cialdini."""
    return await _run_specialist(state, OBJECTION_SYSTEM, temperature=0.6, max_tokens=350)


async def follow_up_node(state: SalesState) -> dict[str, Any]:
    """Retomada de conversa antiga."""
    return await _run_specialist(state, FOLLOW_UP_SYSTEM, temperature=0.7, max_tokens=300)


# ────────────────────────────────────────────────────────────────────
# Nó: summarize (memória longa quando histórico > N)
# ────────────────────────────────────────────────────────────────────

SUMMARIZE_THRESHOLD = 30      # dispara quando há > 30 mensagens
SUMMARIZE_KEEP = 10           # mantém últimas 10 mensagens, resume o resto

SUMMARIZE_SYSTEM = """Resuma a conversa abaixo em até 6 bullets curtos para um vendedor
relembrar rapidamente quem é esse lead, o que ele quer, e onde a conversa parou.

Inclua:
- Nome do cliente (se mencionado)
- Produto/interesse principal
- Objeções já levantadas
- Preço/condição já oferecida
- Próximo passo combinado

NÃO inclua emojis. NÃO faça discurso de vendas. Só fatos."""


async def summarize_node(state: SalesState) -> dict[str, Any]:
    """
    Se messages > SUMMARIZE_THRESHOLD, comprime as antigas em summary.text e
    mantém só as últimas SUMMARIZE_KEEP. Persiste o summary no Redis pra
    reaproveitamento em próximas invocações.
    """
    messages = list(state.get("messages") or [])
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return {}

    cutoff = max(0, len(messages) - SUMMARIZE_KEEP)
    to_summarize = messages[:cutoff]
    recent = messages[cutoff:]

    # Render plain text das mensagens antigas pra entregar ao summarizer.
    rendered_parts: list[str] = []
    if state.get("summary"):
        rendered_parts.append(f"Resumo anterior:\n{state['summary']}")
    for m in to_summarize:
        role = "Cliente" if isinstance(m, HumanMessage) else "Vendedor"
        content = m.content if isinstance(m.content, str) else "[multimodal]"
        rendered_parts.append(f"{role}: {content}")
    rendered = "\n".join(rendered_parts)

    llm = _make_llm(temperature=0.2, max_tokens=400)
    res = await llm.ainvoke([
        SystemMessage(content=SUMMARIZE_SYSTEM),
        HumanMessage(content=rendered),
    ])
    new_summary = (res.content or "").strip()

    if new_summary:
        await get_redis().set_summary(
            instance=state["instance_name"],
            phone=state["phone"],
            summary=new_summary,
        )

    # Trim REAL no checkpointer: emite RemoveMessage por id para cada mensagem antiga.
    # add_messages do LangGraph reconhece RemoveMessage e remove do state.
    # Mensagens sem id (algumas chegam sem) são puladas — o reducer ignora.
    remove_patches: list[Any] = []
    for m in to_summarize:
        msg_id = getattr(m, "id", None)
        if msg_id:
            remove_patches.append(RemoveMessage(id=msg_id))

    return {"summary": new_summary, "messages": remove_patches}


# ────────────────────────────────────────────────────────────────────
# Nó: flow_executor (executa fluxo pré-cadastrado direto via Evolution)
# ────────────────────────────────────────────────────────────────────

_evolution: EvolutionClient | None = None


def get_evolution() -> EvolutionClient:
    """Singleton do cliente Evolution. main.py pode injetar via set_evolution()."""
    global _evolution
    if _evolution is None:
        _evolution = EvolutionClient()
    return _evolution


def set_evolution(client: EvolutionClient) -> None:
    """Injeta um EvolutionClient (útil pra testes com mocks)."""
    global _evolution
    _evolution = client


async def flow_executor_node(state: SalesState) -> dict[str, Any]:
    """
    Dispara a sequência cadastrada pelo nome state["flow_name"]. Marca
    flow_dispatched=True pra send_node pular o envio padrão de chunks.
    """
    import asyncio
    import random

    flow_name = state.get("flow_name")
    if not flow_name:
        return {}

    flow = get_flow(state["project_id"], flow_name)
    if not flow:
        return {"flow_dispatched": False}

    evo = get_evolution()
    instance = state["instance_name"]
    to = state["phone"]
    sent = 0

    for i, step in enumerate(flow.steps):
        if i > 0:
            await asyncio.sleep(1.5 + random.random() * 1.0)
        kind = (step.get("type") or "").lower()
        try:
            if kind == "text":
                await evo.send_typing(instance, to, duration_ms=1200)
                await asyncio.sleep(1.2)
                await evo.send_text(instance, to, step.get("content", ""))
                sent += 1
            elif kind in ("image", "video", "audio", "document"):
                # Envia URL direto (a Evolution baixa do CDN).
                # Para base64 local, plugue uma extensão futura.
                url = step.get("url") or step.get("filePath")
                if not url:
                    continue
                # Reusa send_text como fallback se Evolution não tiver helper:
                # endpoint correto seria /message/sendMedia/<inst> com mediatype.
                body = {
                    "number": to,
                    "mediatype": kind,
                    "media": url,
                    "caption": step.get("caption", ""),
                }
                if kind == "document" and step.get("fileName"):
                    body["fileName"] = step["fileName"]
                # _post é privado; uso public send_text como base de retry e
                # delego via httpx pra path sendMedia.
                async with __import__("httpx").AsyncClient(timeout=evo.timeout) as client:
                    r = await client.post(
                        f"{evo.base_url}/message/sendMedia/{instance}",
                        json=body,
                        headers={"apikey": evo.api_key, "Content-Type": "application/json"},
                    )
                    if r.is_success:
                        sent += 1
        except Exception:  # noqa: BLE001
            # Não derruba o grafo — segue pro próximo step.
            continue

    return {"flow_dispatched": True, "sent": sent > 0, "sent_count": sent}


# ────────────────────────────────────────────────────────────────────
# Nó: tenant_resolver (project_id via Supabase instance_projects)
# ────────────────────────────────────────────────────────────────────

async def tenant_resolver_node(state: SalesState) -> dict[str, Any]:
    """
    Se project_id já veio (query string ou injetado), respeita.
    Senão, busca em instance_projects via Supabase. Fallback: DEFAULT_PROJECT_ID.
    """
    if state.get("project_id"):
        return {}

    resolver = get_tenant_resolver()
    project_id = await resolver.resolve(state["instance_name"])
    if not project_id:
        project_id = os.getenv("DEFAULT_PROJECT_ID", "padrao")
    return {"project_id": project_id}


# ────────────────────────────────────────────────────────────────────
# Nó: send (envia bolhas no WhatsApp com typing e jitter)
# ────────────────────────────────────────────────────────────────────

async def send_node(state: SalesState) -> dict[str, Any]:
    """
    Envia as bolhas geradas por respond/close/specialists via Evolution API.
    Aplica typing simulation + jitter entre bolhas + reação opcional.

    Pula se:
      - flow_dispatched=True (flow_executor_node já enviou tudo)
      - chunks vazio
    """
    import asyncio

    from agent.tools import jitter_between_bubbles_ms, typing_delay_ms

    if state.get("flow_dispatched"):
        return {}

    chunks = state.get("chunks") or []
    if not chunks:
        return {"sent": False, "sent_count": 0}

    evo = get_evolution()
    instance = state["instance_name"]
    phone = state["phone"]
    message_id = state.get("message_id", "")

    # Reação opcional (antes das bolhas, igual ao bot antigo)
    react_emoji = state.get("react_emoji")
    if react_emoji and message_id:
        try:
            await evo.send_reaction(instance, phone, message_id, react_emoji)
        except Exception:  # noqa: BLE001
            pass  # reação é cosmético; não falha o envio

    sent = 0
    for i, chunk in enumerate(chunks):
        delay = typing_delay_ms(chunk)
        try:
            await evo.send_typing(instance, phone, duration_ms=delay)
            await asyncio.sleep(delay / 1000)
            r = await evo.send_text(instance, phone, chunk)
            if r.get("success"):
                sent += 1
            if i < len(chunks) - 1:
                await asyncio.sleep(jitter_between_bubbles_ms() / 1000)
        except Exception:  # noqa: BLE001
            # Não derruba o grafo — registra parcial e segue.
            continue

    return {"sent": sent > 0, "sent_count": sent}
