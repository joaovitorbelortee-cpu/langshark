"""
Nós do grafo de vendas LangGraph.

Cada função é um nó: recebe o SalesState, devolve um patch (dict) com os campos
que mudaram. LangGraph faz o merge — em particular, `messages` é appended via
add_messages.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import Intent, SalesState
from agent.tools import chunk_for_whatsapp, parse_tags
from memory.redis_store import RedisStore
from rag.catalog import CatalogRAG


# ────────────────────────────────────────────────────────────────────
# Singletons (criados uma vez por processo)
# ────────────────────────────────────────────────────────────────────

_redis: RedisStore | None = None
_rag: CatalogRAG | None = None


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


def _build_system_prompt(catalog_block: str) -> str:
    return SALES_SYSTEM + ("\n\n" + catalog_block if catalog_block else "")


# ────────────────────────────────────────────────────────────────────
# Nó: load_history
# ────────────────────────────────────────────────────────────────────

async def load_history_node(state: SalesState) -> dict[str, Any]:
    """Carrega histórico do Redis e adiciona a mensagem atual do usuário."""
    history = await get_redis().load_history(
        instance=state["instance_name"],
        phone=state["phone"],
    )
    user_msg = HumanMessage(content=state.get("user_message") or "[mídia]")
    return {"messages": [*history, user_msg]}


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
    catalog_block = get_rag().format_context(state.get("catalog_hits", []) or [])
    system_prompt = _build_system_prompt(catalog_block)

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    messages.extend(state.get("messages", []))

    llm = _make_llm(temperature=0.7, max_tokens=600)
    res = await llm.ainvoke(messages)
    raw = res.content or ""

    parsed = parse_tags(raw)
    chunks = chunk_for_whatsapp(parsed.text, max_bubbles=2, max_chars=320)

    return {
        "reply": parsed.text,
        "chunks": chunks,
        "has_converted": parsed.has_converted,
        "schedule_minutes": parsed.schedule_minutes,
        "react_emoji": parsed.react_emoji,
        "quote_previous": parsed.quote_previous,
        "messages": [AIMessage(content=parsed.text)],
    }


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
    system = CLOSE_SYSTEM + ("\n\n" + catalog_block if catalog_block else "")

    messages: list[Any] = [SystemMessage(content=system)]
    messages.extend(state.get("messages", []))

    llm = _make_llm(temperature=0.5, max_tokens=400)
    res = await llm.ainvoke(messages)
    parsed = parse_tags(res.content or "")
    chunks = chunk_for_whatsapp(parsed.text, max_bubbles=2, max_chars=320)

    return {
        "reply": parsed.text,
        "chunks": chunks,
        "has_converted": parsed.has_converted,
        "schedule_minutes": parsed.schedule_minutes,
        "react_emoji": parsed.react_emoji,
        "quote_previous": parsed.quote_previous,
        "messages": [AIMessage(content=parsed.text)],
    }


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
