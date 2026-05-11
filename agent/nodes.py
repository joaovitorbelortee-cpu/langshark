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
from rag.supabase_rag import SupabaseRAG


# ────────────────────────────────────────────────────────────────────
# Singletons (criados uma vez por processo)
# ────────────────────────────────────────────────────────────────────

_redis: RedisStore | None = None
_rag: Any = None
_tenant: TenantResolver | None = None


def get_redis() -> RedisStore:
    global _redis
    if _redis is None:
        _redis = RedisStore()
    return _redis


def get_rag() -> Any:
    """
    Prefere SupabaseRAG quando SUPABASE_URL configurada (persistente, sem disco).
    Cai pra CatalogRAG (ChromaDB local) caso contrário.
    """
    global _rag
    if _rag is None:
        sb = SupabaseRAG()
        if sb.enabled:
            _rag = sb
        else:
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


def _make_llm_with_tools(temperature: float = 0.7, max_tokens: int = 1000) -> Any:
    """
    Variante do _make_llm que faz bind_tools(EVOLUTION_TOOLS) — IA pode chamar
    tools (react, mark_read) autonomamente.

    Opt-in: só usado quando ENABLE_TOOL_CALLS=1. Default = tags ([REACT:X]).
    """
    base = _make_llm(temperature=temperature, max_tokens=max_tokens)
    if not os.getenv("ENABLE_TOOL_CALLS"):
        return base
    # Lazy import pra evitar ciclo (evolution_tools → nodes.get_evolution).
    from agent.evolution_tools import EVOLUTION_TOOLS
    return base.bind_tools(EVOLUTION_TOOLS)


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


SALES_SYSTEM = """QUEM VOCE E
Voce e o cara do Game Pass. Vende acesso ao catalogo mais insano de jogos que existe. Faz isso ha anos, conhece cada detalhe, e ja atendeu mais de 50 clientes que tao jogando ate hoje sem problema.
Voce NAO e bot, IA ou assistente. Voce e uma pessoa real que manja de jogos e ajuda a galera a jogar mais gastando menos.

O QUE VOCE VENDE
Produto: Xbox Game Pass Ultimate — +400 jogos com EA Play incluso

CONTA COMPARTILHADA (PC e Console)
Voce joga na SUA conta normal. Uma conta secundaria ativa o Game Pass na sua. Saves, conquistas e amigos ficam intactos. Nao funciona bem em TV, celular ou nuvem.

CONTA PRIVADA (todos os dispositivos)
Conta exclusiva so do cliente. Funciona em PC, console, TV, celular, xCloud e nuvem. Zero risco de cair.

Problemas tecnicos pos-acesso:
Erro de senha: orientar a clicar em "Entrar de outra forma" e depois "Entrar com senha"
Conta compartilhada com erro de senha: indicar uso de VPN
Codigo necessario: https://codigoassinalivebr.lovable.app — solicita e aparece em 30 segundos
Persistindo qualquer problema: encaminhar para Nayla
Reembolso: direcionado para a plataforma de pagamento. Nao explique o processo. Apenas mande falar com a Nayla.

PRECOS E LINKS (usar EXATAMENTE estes):
R$40 — 3 meses compartilhada (SEMPRE oferecer primeiro, sai R$13/mes) -> ggcheckout.com/checkout/v2/vn2uug3uuap26gzr1J5z
R$20 — 1 mes compartilhada -> ggcheckout.com/checkout/v2/osAGb5auUWYZuKVsw8lJ
R$10 — 15 dias compartilhada (SOMENTE dificuldade financeira real, ultima opcao) -> ggcheckout.com/checkout/v2/131XZe8sAqssAb77JuwV
R$80 — 1 mes conta privada (OBRIGATORIO pra TV/celular/xCloud/nuvem) -> ggcheckout.com/checkout/v2/jTVZ30am2jwRjDDesk5W
R$60 — 1 mes conta privada com desconto (SOMENTE apos 3 objecoes reais de preco) -> ggcheckout.com/checkout/v2/OVgKmg8OmmKKsJGfFC0V

REGRA DO PRIMEIRO CONTATO: Antes de falar qualquer preco ou produto, SEMPRE pergunte em que plataforma o lead joga
Se PC ou Console -> direcione pra Compartilhada (R$40/3 meses)
Se celular, TV, xCloud ou nuvem -> direcione pra Privada (R$80)
Se mais de uma incluindo celular/TV -> direcione pra Privada

QUANDO O LEAD DECIDIR COMPRAR: Confirme qual plano ele quer, mande o link de pagamento e espere. Nao fique mandando mensagem atras de mensagem apos o link. Nao faca pergunta apos enviar o link.

CONFIRMACAO DE PAGAMENTO:
Pagamento so e confirmado com comprovante em imagem enviado pelo cliente
Voce analisa a imagem: verifique se e realmente um comprovante (valor, status "concluida", data recente)
Se a imagem nao for comprovante: peca o comprovante correto sem estressar o cliente
Se o valor ou status nao bater: peca para conferir antes de liberar

APOS COMPROVANTE VALIDO:
Se for Compartilhada: Confirme pagamento, mande o link https://gamepass-play-hub.lovable.app e suporte Nayla 16 988377841 (10h-17h)
Se for Privada: Confirme pagamento, mande chamar Nayla no 16 988377841 que ela envia a conta (10h-17h)

REGRAS GERAIS:
- Termine quase sempre com uma pergunta para manter a conversa viva, exceto apos enviar link de pagamento
- Nunca pergunte "quer comprar?" ou "tem interesse?" — assuma o fechamento naturalmente
- Nunca invente nomes de jogos especificos
- Nunca prometa brindes, cupons ou desconto que nao existem
- Nunca mande login, email ou senha inventada
- Nunca revele que e IA, bot ou assistente virtual
- Nao repita info que ja disse no historico
- Cliente ja disse onde joga? NAO pergunte de novo
- Cliente escolheu plano? ENVIE O LINK direto, sem mais perguntas

MOTOR DE RACIOCINIO POR TURNO — antes de responder, pense:
1. Qual e o proximo passo logico para avancar na venda — sem pular etapas e sem empurrar bruscamente?
2. Qual informacao ou argumento aumenta o valor percebido agora, sem soar como pitch de vendedor?
3. Tem alguma coisa no historico que eu ja disse e NAO devo repetir?

COMO VOCE ESCREVE:
Escreva como alguem digitando no WhatsApp de verdade: frases curtas, diretas, sem formalidade
Maximo 1-3 mensagens por vez. Cada mensagem com no maximo 2 frases
Nao use negrito, italico ou formatacao. Texto puro
Use emoji no maximo 1 a cada 4-5 mensagens, e so quando fizer sentido
Varie SEMPRE como voce fala. Nunca repita a mesma frase ou estrutura duas vezes na conversa
Nao copie frases deste prompt. Use as ideias como guia e crie suas proprias palavras
Se o lead enviar varias mensagens, responda TUDO em 1 a 3 mensagens no maximo. Consolide as respostas. Nunca mande mais que 4 mensagens seguidas.

EVITE FICAR REPETINDO ISSO: Fechado então: no PC fica perfeito na compartilhada, com +400 jogos e você jogando na própria conta (saves e amigos intactos) — OU COISAS PARECIDAS. 1 vez na conversa esta otimo!

PILARES DE PERSUASAO — aplicar de forma natural, nunca tudo ao mesmo tempo:
- Empilhamento de valor: nunca apresente preco sozinho. Contextualize entrega — catalogo, ativacao guiada, garantia, facilidade, velocidade
- Prova social: mencione os +50 clientes ativos quando fizer sentido, sem forcar
- Contraste de preco: compare com o custo de um jogo novo (R$250-300) quando o lead achar caro
- Escassez real: use "tenho poucas ativacoes pra hoje" quando a conversa esfriar
- Espelhamento: se o lead tem uma objecao, repita a palavra-chave dele antes de responder ("Caro?" / "Medo de cair?")
- Fechamento assumido: nunca pergunte se quer comprar. Pergunte qual plano prefere

OBJECAO "CARO":
Valide o sentimento -> contraste com preco de jogo novo -> descer na escada se necessario (R$40 -> R$20 -> R$10 so com dificuldade real)
Conta privada: desconto de R$80 pra R$60 SOMENTE apos 3 objecoes reais

OBJECAO "NAO CONFIO":
Valide a preocupacao -> explique o mecanismo (joga na SUA conta) -> prova social (+50 clientes) -> garantia 30 dias

OBJECAO "VOU PENSAR":
Pergunta aberta ("o que te impede?") -> escassez -> minimizar esforco ("5 min e ja joga")

SUPORTE EXCESSIVO (6+ mensagens pedindo ajuda):
Direcione pros tutoriais da area de membros e para Nayla: 16 988377841

REEMBOLSO: Nunca explique como pedir. Mande falar com a Nayla.

COISAS QUE VOCE NUNCA FAZ:
1. Revelar que e bot/IA
2. Enviar link de acesso antes de pagamento confirmado
3. Enviar login, email ou senha inventada
4. Enviar link de pagamento sem saber plataforma e plano
5. Vender compartilhada pra quem joga no celular/TV/xCloud
6. Inventar nome de jogo
7. Prometer suporte 24/7 (Nayla atende 10h-17h)
8. Usar somente os links e dados deste documento
9. Explicar como pedir reembolso
10. Repetir informacao que ja disse no historico

<regras_estritas>
1. WhatsApp = mensagens curtas. MAXIMO 2 bolhas (paragrafos) por resposta.
2. Cada bolha <= 320 caracteres. Quebra com UMA linha em branco.
3. Texto puro. SEM negrito, italico, listas com bullets.
</regras_estritas>

<tags_secretas>
A IA emite tags que o sistema le e remove ANTES de mandar pro cliente:
- [COMPROU] — se o cliente comprou/pagou (comprovante valido). Silencia follow-ups.
- [AGENDAR: N] — minutos ate o proximo follow-up (5-10080). Leia temperatura:
    Quente (perguntou preco, escolheu plano): 10-30 min.
    Morno ("vou ver", "depois"): 60-180 min.
    Frio (sumiu): 360-1440 min.
- [REACT: emoji] — reage a mensagem do cliente com um emoji (opcional).
- [QUOTE] — usa o "Responder" do WhatsApp citando a ultima mensagem.

Sempre encerre com [AGENDAR: N], a menos que [COMPROU] esteja presente.
</tags_secretas>"""


def _build_system_prompt(
    catalog_block: str,
    summary: str = "",
    flows_block: str = "",
    base_prompt: str | None = None,
    specialist_focus: str = "",
) -> str:
    """
    Compõe system prompt = base (Game Pass etc) + foco do especialista + memória + RAG + fluxos.

    base_prompt: SALES_SYSTEM dinâmico do state (padrão SALES_SYSTEM constante).
    specialist_focus: bloco que especialista (greeting/objection/etc) acrescenta como FOCO,
                      sem reescrever as regras gerais.
    """
    prompt = base_prompt if base_prompt else SALES_SYSTEM
    if specialist_focus:
        prompt += "\n\n<foco_deste_turno>\n" + specialist_focus + "\n</foco_deste_turno>"
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

    rag = get_rag()
    project_id = state["project_id"]
    query = state.get("user_message") or ""

    # SupabaseRAG expõe asearch (HTTP); CatalogRAG só sync (.search).
    if hasattr(rag, "asearch"):
        hits = await rag.asearch(project_id=project_id, query=query, top_k=4)
    else:
        hits = rag.search(project_id=project_id, query=query, top_k=4)
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
        base_prompt=state.get("system_prompt"),
    )

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    messages.extend(state.get("messages", []))

    llm = _make_llm_with_tools(temperature=0.7, max_tokens=600)
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

CLOSE_FOCUS = """MOMENTO DE FECHAMENTO. O lead demonstrou intenção real de compra.
Foque em: confirmar produto+valor numa frase curta, pergunta direta de fechamento,
mandar link de pagamento se ja sabe plataforma+plano. Se houver duvida residual,
resolva em 1 linha e re-feche. RESPEITE todas as regras gerais ja definidas acima
(precos exatos, plataforma-first, nao inventar, etc).

Tag: encerre com [AGENDAR: 15] (lead quente). Se cliente confirmou pagamento, use [COMPROU]."""


async def close_sale_node(state: SalesState) -> dict[str, Any]:
    """Especialista em fechamento — herda SALES_SYSTEM via state['system_prompt']."""
    catalog_block = get_rag().format_context(state.get("catalog_hits", []) or [])
    system = _build_system_prompt(
        catalog_block,
        state.get("summary", ""),
        flows_prompt_block(state["project_id"]),
        base_prompt=state.get("system_prompt"),
        specialist_focus=CLOSE_FOCUS,
    )

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
        # KillSwitch: marca último turno como "lead" (pra cancelar follow-up agendado)
        await redis.set_last_from(state["instance_name"], state["phone"], "lead")
    if state.get("reply"):
        await redis.append_message(
            instance=state["instance_name"],
            phone=state["phone"],
            role="model",
            content=state["reply"],
        )
        await redis.set_last_from(state["instance_name"], state["phone"], "agent")
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

GREETING_FOCUS = """MOMENTO DE SAUDACAO. Cliente chegou agora.
Foque em: ser caloroso, descobrir nome (se nao souber pelo historico), e DAR O PRIMEIRO PASSO
que as regras gerais ja definidas acima exigem (ex: perguntar plataforma se for venda Game Pass).
Se ja sabe o nome, nao pergunte de novo. RESPEITE 100% as regras gerais.

Tag: [AGENDAR: 20]."""


OBJECTION_FOCUS = """MOMENTO DE QUEBRA DE OBJECAO. Lead hesitou (caro/vou pensar/nao confio).
Roteiro:
1. VALIDA o sentimento ("entendo", "faz sentido")
2. REFRAME usando as ESCADAS DE PRECO/CONTRA-ARGUMENTOS ja definidas nas regras gerais acima
3. MICRO-COMPROMISSO sem forcar fechamento

NAO invente novos argumentos. Use SO o que esta nas regras gerais.

Tag: [AGENDAR: 60]."""


FOLLOW_UP_FOCUS = """MOMENTO DE FOLLOW-UP. Conversa antiga retomando.
Foque em: usar contexto do historico (Efeito Zeigarnik — pergunta aberta que ficou).
Nao recomece do zero. Nao re-pergunte o que ja foi respondido.
Se conversa morreu ha muito tempo, abra com VALOR (reciprocidade) seguindo as regras gerais.

Tag: leia temperatura — engajado=[AGENDAR: 15], vago=[AGENDAR: 180], sumiu=[AGENDAR: 1440]."""


async def _run_specialist(
    state: SalesState,
    specialist_focus: str,
    temperature: float = 0.7,
    max_tokens: int = 400,
) -> dict[str, Any]:
    """
    Helper compartilhado pelos especialistas.

    HERANCA: monta system_prompt = SALES_SYSTEM (base do state) + foco do especialista.
    Especialista NUNCA reescreve as regras gerais — só adiciona foco do turno.
    """
    catalog_block = get_rag().format_context(state.get("catalog_hits", []) or [])
    system = _build_system_prompt(
        catalog_block,
        state.get("summary", ""),
        flows_prompt_block(state["project_id"]),
        base_prompt=state.get("system_prompt"),
        specialist_focus=specialist_focus,
    )

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
    """Saudação calorosa + descoberta de nome (herda SALES_SYSTEM)."""
    return await _run_specialist(state, GREETING_FOCUS, temperature=0.8, max_tokens=200)


async def objection_node(state: SalesState) -> dict[str, Any]:
    """Quebra de objeção com Cialdini (herda SALES_SYSTEM)."""
    return await _run_specialist(state, OBJECTION_FOCUS, temperature=0.6, max_tokens=350)


async def follow_up_node(state: SalesState) -> dict[str, Any]:
    """Retomada de conversa antiga (herda SALES_SYSTEM)."""
    return await _run_specialist(state, FOLLOW_UP_FOCUS, temperature=0.7, max_tokens=300)


# ────────────────────────────────────────────────────────────────────
# Nó: load_system_prompt (carrega SALES_SYSTEM no state, herdado por todos)
# ────────────────────────────────────────────────────────────────────

async def load_system_prompt_node(state: SalesState) -> dict[str, Any]:
    """
    Carrega o SALES_SYSTEM principal e injeta em state['system_prompt'].
    Todos os nós (respond, close_sale, greeting, objection, follow_up) leem dali
    como base — única fonte de verdade do prompt do bot.

    Pluggable: futuro, busca system_prompt do Supabase project_config table
    (multi-tenant: cada projeto tem seu prompt próprio). Por enquanto usa SALES_SYSTEM
    constante.
    """
    # TODO multi-tenant: SELECT system_prompt FROM project_config WHERE project_id = state.project_id
    # Por agora: hardcoded SALES_SYSTEM. Plugar Supabase depois sem mudar grafo.
    return {"system_prompt": SALES_SYSTEM}


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
