"""
Nós do grafo de vendas LangGraph.

Cada função é um nó: recebe o SalesState, devolve um patch (dict) com os campos
que mudaram. LangGraph faz o merge — em particular, `messages` é appended via
add_messages.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI

log = logging.getLogger("agent.nodes")

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


def _llm_timeout_seconds() -> float:
    """Timeout duro pra chamada LLM. Evita webhook pendurado se provider travar."""
    try:
        return float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    except (TypeError, ValueError):
        return 30.0


def _make_llm(
    temperature: float = 0.7,
    max_tokens: int = 1000,
    model_override: str | None = None,
) -> ChatOpenAI:
    """
    Constrói o cliente LLM. model_override > AI_MODEL env > default.
    Permite trocar de modelo por projeto via Supabase project_config sem redeploy.
    """
    return ChatOpenAI(
        model=model_override or os.getenv("AI_MODEL", "openai/gpt-4o-mini"),
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "missing",
        base_url=os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=_llm_timeout_seconds(),
        max_retries=2,
        default_headers={
            "HTTP-Referer": os.getenv("AI_REFERRER", "https://bot-vendas.local"),
            "X-Title": "bot-vendas",
        },
    )


def _make_llm_with_tools(
    temperature: float = 0.7,
    max_tokens: int = 1000,
    model_override: str | None = None,
) -> Any:
    """
    Variante do _make_llm que faz bind_tools(EVOLUTION_TOOLS) — IA pode chamar
    tools (react, mark_read) autonomamente.

    Opt-in: só usado quando ENABLE_TOOL_CALLS=1. Default = tags ([REACT:X]).
    """
    base = _make_llm(temperature=temperature, max_tokens=max_tokens, model_override=model_override)
    if not os.getenv("ENABLE_TOOL_CALLS"):
        return base
    # Lazy import pra evitar ciclo (evolution_tools → nodes.get_evolution).
    from agent.evolution_tools import EVOLUTION_TOOLS
    return base.bind_tools(EVOLUTION_TOOLS)


def _llm_kwargs_from_state(state: SalesState, default_temp: float = 0.7, default_max: int = 600) -> dict[str, Any]:
    """Lê overrides de state (preenchidos por load_system_prompt_node)."""
    return {
        "temperature": float(state.get("_ai_temperature") or default_temp),
        "max_tokens": int(state.get("_ai_max_tokens") or default_max),
        "model_override": state.get("_ai_model"),
    }


# ────────────────────────────────────────────────────────────────────
# Prompts
# ────────────────────────────────────────────────────────────────────

INTENT_PROMPT = """Você classifica a INTENÇÃO da última mensagem do cliente em UMA das categorias.

REGRA CRÍTICA: Classifique baseado no CONTEXTO da conversa, não apenas na frase isolada.
Uma mesma frase pode ter intenção diferente dependendo do estágio.
Exemplos:
  - "oi" SEM histórico = saudacao
  - "oi" COM histórico de discussão de preço = follow_up (retomada)
  - "ta caro" no início = objecao
  - "ta caro" depois do bot dar 2 descontos = intencao_compra com hesitação suave

Categorias:
- saudacao: PRIMEIRO contato. Cliente cumprimenta SEM histórico prévio na conversa.
- duvida_produto: pergunta sobre produto, características, como funciona, qual diferença
- pedir_preco: quer saber valor, quanto custa, formas de pagamento, link
- objecao: hesitação real, "tá caro", "vou pensar", "depois eu vejo", "preciso conversar"
- intencao_compra: quer fechar, "vou pegar", "pode mandar link", escolheu plano
- comprou: pagou, mandou comprovante, "fechei", "paguei"
- follow_up: cliente retomando conversa antiga ("eae", "oi", "voltei") JÁ COM HISTÓRICO
- outros: continuação trivial, confirmação curta ("ok", "valeu", "blz", emoji)

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
- [COMPROU] — cliente pagou (comprovante valido). Silencia follow-ups.
- [AGENDAR: N] — minutos ate o proximo follow-up (5-10080).
    Quente=10-30, Morno=60-180, Frio=360-1440.
- [REACT: emoji] — bot reage com 1 emoji. PARCIMONIA: 20-30% das msgs.
    Use SO quando: engraçada(😂), triste(😢), empolgante(🔥), surpresa(😮),
    agradecimento(🙏), confirmou compra(🎉). Nunca 2 seguidas.
- [QUOTE] — bot cita msg do cliente (Responder WhatsApp).
    AUTOMATICO quando msg tem "?". Use manual em: mudanca de tema,
    multiplas perguntas/afirmacoes, contexto necessario. Nunca 2 seguidas.

Sempre encerre com [AGENDAR: N], a menos que [COMPROU] esteja presente.
</tags_secretas>"""


def _build_system_prompt(
    catalog_block: str,
    summary: str = "",
    flows_block: str = "",
    base_prompt: str | None = None,
    specialist_focus: str = "",
    lead_facts: dict[str, Any] | None = None,
    supervisor_feedback: str | None = None,
) -> str:
    """
    Compõe system prompt = base + foco + lead_facts (CRÍTICO anti-amnésia) +
    summary + RAG + flows + supervisor_feedback (se retry).

    lead_facts: dict estruturado {plataforma, nome, plano_interesse, objecoes, estagio, ...}
                Renderizado como bloco <lead_conhecido> que instrui bot a NÃO
                reperguntar dados conhecidos.
    supervisor_feedback: instrução de correção do supervisor quando especialista
                        precisa refazer resposta rejeitada.
    """
    prompt = base_prompt if base_prompt else SALES_SYSTEM

    # Lead facts vem ANTES do specialist_focus pra prevenir override do prompt.
    if lead_facts:
        from agent.lead_memory import format_for_prompt
        facts_block = format_for_prompt(lead_facts)
        if facts_block:
            prompt += "\n\n" + facts_block

    if specialist_focus:
        prompt += "\n\n<foco_deste_turno>\n" + specialist_focus + "\n</foco_deste_turno>"

    # Supervisor feedback vai PRÓXIMO do final pra ter mais peso no LLM
    if supervisor_feedback:
        prompt += (
            "\n\n<supervisor_feedback>\n"
            "ATENÇÃO: sua resposta anterior foi REJEITADA pelo supervisor.\n"
            f"Motivo/correção: {supervisor_feedback}\n"
            "REESCREVA seguindo essa instrução. NÃO repita o erro.\n"
            "</supervisor_feedback>"
        )

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
    """
    Classifica intenção da última mensagem CONSIDERANDO contexto.

    LLM recebe histórico recente + última msg pra decidir. Sem isso,
    "eae" depois de 5 turnos vira "saudacao" falsamente → bot recomeça.
    """
    user_msg = state.get("user_message") or ""
    if not user_msg.strip():
        return {"intent": "outros"}

    # Pega últimas 12 msgs do histórico (excluindo a current — última no array)
    msgs = state.get("messages") or []
    # state["messages"] inclui a current user_msg como última. Pega anteriores.
    prior = msgs[:-1][-12:] if len(msgs) > 1 else []

    convo_lines: list[str] = []
    for m in prior:
        msg_type = getattr(m, "type", "")
        role = "agent" if msg_type == "ai" else ("cliente" if msg_type == "human" else msg_type)
        text = getattr(m, "content", "")
        if isinstance(text, str) and text.strip():
            convo_lines.append(f"{role}: {text[:220]}")
    history_block = "\n".join(convo_lines) if convo_lines else "(sem histórico — primeiro contato)"

    user_prompt = (
        f"HISTÓRICO RECENTE DA CONVERSA:\n{history_block}\n\n"
        f"ÚLTIMA MENSAGEM DO CLIENTE:\n{user_msg}\n\n"
        f"Classifique a intenção CONSIDERANDO o contexto acima. "
        f"Se houver histórico substantivo, NUNCA classifique como 'saudacao'."
    )

    llm = _make_llm(temperature=0.0, max_tokens=15)
    res = await llm.ainvoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    raw = (res.content or "").strip().lower().split()[0] if res.content else "outros"
    raw = raw.strip(".,;:!?")

    valid: tuple[Intent, ...] = (
        "saudacao", "duvida_produto", "pedir_preco",
        "objecao", "intencao_compra", "comprou", "follow_up", "outros",
    )
    intent: Intent = raw if raw in valid else "outros"

    # Salvaguarda redundante: se classificou saudacao mas JÁ HOUVE qualquer turno
    # do bot, força follow_up. Anti-amnésia agressivo — 1 msg do bot = conversa
    # iniciada, não saudação.
    bot_already_spoke = any(getattr(m, "type", "") == "ai" for m in prior)
    if intent == "saudacao" and bot_already_spoke:
        log.info("[intent] downgrade saudacao→follow_up: bot já falou no histórico")
        intent = "follow_up"

    return {"intent": intent}


# ────────────────────────────────────────────────────────────────────
# Nó: lead_memory (extrai fatos estruturados pra anti-amnésia)
# ────────────────────────────────────────────────────────────────────

async def lead_memory_node(state: SalesState) -> dict[str, Any]:
    """
    Mantém estado estruturado do lead (plataforma/plano/estágio/objeções) em Redis.

    Roda em TODO turn inbound. Lê fatos existentes, atualiza com histórico atual
    via LLM extraction leve, salva, injeta no state pra especialistas usarem.

    Pula em follow_up turn (bot iniciando — sem nova msg do cliente pra extrair).
    """
    if state.get("intent") == "follow_up":
        return {}

    from agent.lead_memory import empty_facts, extract_facts

    redis = get_redis()
    instance = state.get("instance_name", "")
    phone = state.get("phone", "")

    # Lê facts existentes (lead pode já ter histórico de turnos passados)
    try:
        current_facts = await redis.get_lead_facts(instance, phone) or empty_facts()
    except Exception:  # noqa: BLE001
        current_facts = empty_facts()

    # Atualiza via LLM extraction com histórico ATÉ AGORA
    try:
        new_facts = await extract_facts(
            messages=state.get("messages") or [],
            current_facts=current_facts,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[lead_facts] extract erro: %s — mantém current", exc)
        new_facts = current_facts

    # Persiste apenas se mudou (evita writes inúteis)
    try:
        if new_facts != current_facts:
            await redis.set_lead_facts(instance, phone, new_facts)
            log.info(
                "[lead_facts] %s/%s update: plataforma=%s estagio=%s plano=%s",
                instance, phone,
                new_facts.get("plataforma"),
                new_facts.get("estagio"),
                new_facts.get("plano_interesse"),
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("[lead_facts] persist erro: %s", exc)

    return {"lead_facts": new_facts}


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
        lead_facts=state.get("lead_facts"),
        supervisor_feedback=state.get("supervisor_feedback"),
    )

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    messages.extend(state.get("messages", []))

    llm = _make_llm_with_tools(**_llm_kwargs_from_state(state, default_temp=0.7, default_max=600))
    res = await llm.ainvoke(messages)
    raw = res.content or ""

    parsed = parse_tags(raw)
    # Remove a tag [FLOW: nome] do texto antes de chunkar.
    flow_name, cleaned_text = parse_flow_tag(parsed.text)
    chunks = chunk_for_whatsapp(cleaned_text, max_bubbles=4, max_chars=110)

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

CLOSE_FOCUS = """FECHAMENTO — lead quer comprar. PENSE no momento exato e responda.

NAO USE FRASES PRONTAS. Cada fechamento e diferente — adapte ao lead.

ANTES de escrever, PENSE:
- Qual plataforma ele tem? (lead_conhecido.plataforma)
- Qual plano ele perguntou ou voce ofereceu? (plano_interesse)
- Ele ja recebeu link? Se sim, REFORCE link existente nao mande novo.
- Ele tem alguma duvida residual? Resolva em 1 linha curta.

PRINCIPIOS:
- Confirme valor+produto em 1 frase clara (sem ambiguidade).
- Pergunta de fechamento direta mas natural (nao templated "vamos fechar?").
- Mande o link CERTO baseado no que ele escolheu.

JAMAIS:
- Mude o preco que ja foi acordado.
- Mande link errado de plataforma.
- Use frase batida tipo "vamos fechar essa?".
- Invente preco novo.

Tag: [AGENDAR: 15] (lead quente). [COMPROU] se confirmou pagamento."""


async def close_sale_node(state: SalesState) -> dict[str, Any]:
    """Especialista em fechamento — herda SALES_SYSTEM via state['system_prompt']."""
    catalog_block = get_rag().format_context(state.get("catalog_hits", []) or [])
    system = _build_system_prompt(
        catalog_block,
        state.get("summary", ""),
        flows_prompt_block(state["project_id"]),
        base_prompt=state.get("system_prompt"),
        specialist_focus=CLOSE_FOCUS,
        lead_facts=state.get("lead_facts"),
        supervisor_feedback=state.get("supervisor_feedback"),
    )

    messages: list[Any] = [SystemMessage(content=system)]
    messages.extend(state.get("messages", []))

    llm = _make_llm(**_llm_kwargs_from_state(state, default_temp=0.5, default_max=400))
    res = await llm.ainvoke(messages)
    parsed = parse_tags(res.content or "")
    flow_name, cleaned_text = parse_flow_tag(parsed.text)
    chunks = chunk_for_whatsapp(cleaned_text, max_bubbles=4, max_chars=110)

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

GREETING_FOCUS = """SAUDACAO / RETOMADA — analise CONTEXTO antes de qualquer palavra.

NAO USE MENSAGENS PRONTAS. Pense no estagio da conversa, no que o lead falou,
no que voce ja sabe (lead_conhecido) e ESCREVA do zero algo natural.

PRINCIPIOS (nao templates):
- Cumprimente proporcionalmente: muito casual com lead casual; mais formal se ele e.
- Se ja sabe alguma coisa do lead (plataforma/nome/plano), USE essa informacao
  pra mostrar que voce lembra — sem repetir literalmente, integre na fala.
- Avance no estagio da conversa. NUNCA volte a estagios passados.
- Se cliente demonstrou confusao, peça desculpa BREVE e continue de onde parou.
- Se ele ficou em silencio e voltou, retome com algo concreto do contexto, NUNCA
  generico tipo "como posso te ajudar?".
- Use o ULTIMO_RESUMO em lead_conhecido pra calibrar onde a conversa parou.

JAMAIS faca:
- Pergunta que ja foi respondida ("qual plataforma?" se ja sabe).
- Apresentacao genérica ("Posso te ajudar?" sem contexto).
- Pitch de produto se voce ja apresentou.
- Frases tipicas de bot ("Como posso te ajudar hoje?").

Tag: [AGENDAR: 20]."""


OBJECTION_FOCUS = """OBJECAO — analise QUAL e a hesitacao real do lead e responda especifico.

NAO USE FRASES PRONTAS. Pense:
- Qual e a objecao exata (caro, tempo, confianca, alguem decide com ele...)?
- Ela ja foi tratada antes (lead_conhecido.objecoes)? Se sim, NAO repita argumento.
- Que angulo NOVO voce pode trazer baseado no que sabe do lead?

PRINCIPIOS:
1. RECONHECA o sentimento dele com palavras dele (nao "entendo", muito generico).
2. REFRAME com argumento das regras gerais — mas adaptado ao contexto especifico.
3. PERGUNTA SUAVE de micro-compromisso (nao force fechamento).

JAMAIS:
- Invente argumento que nao esta nas regras gerais.
- Use template tipo "entendo seu lado, mas...".
- Repita argumento ja tratado nesta conversa.

Tag: [AGENDAR: 60]."""


FOLLOW_UP_FOCUS = """FOLLOW-UP / RETOMADA — bot esta iniciando contato apos pausa.

NAO USE FRASES PRONTAS. Bot real sabe puxar assunto de onde parou.

ANTES de escrever, PENSE:
- Qual era o ULTIMO assunto (ultimo_resumo em lead_conhecido)?
- O lead estava em qual estagio?
- Que pergunta aberta ficou sem resposta (Zeigarnik)?
- Quanto tempo passou? (se muito, abra com algo de valor antes de pedir)

CADA mensagem deve ser UNICA — escrita pensando NESTE lead. Nada generico.

JAMAIS:
- "Oi, voce ainda esta interessado?" (genérico, bot).
- "Como posso te ajudar?" (sem contexto).
- Recomeçar pitch.
- Repetir mesma abordagem do follow-up anterior.

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
        lead_facts=state.get("lead_facts"),
        supervisor_feedback=state.get("supervisor_feedback"),
    )

    messages: list[Any] = [SystemMessage(content=system)]
    messages.extend(state.get("messages", []))

    kwargs = _llm_kwargs_from_state(state, default_temp=temperature, default_max=max_tokens)
    llm = _make_llm(**kwargs)
    res = await llm.ainvoke(messages)
    parsed = parse_tags(res.content or "")
    flow_name, cleaned_text = parse_flow_tag(parsed.text)
    chunks = chunk_for_whatsapp(cleaned_text, max_bubbles=4, max_chars=110)

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
    Carrega config do projeto (Supabase project_config) via cache LRU TTL 60s.

    Compõe system_prompt a partir das 5 seções editáveis (brain_sections) +
    footer técnico (tags secretas, regras estritas). Fallback pra SALES_SYSTEM
    hardcoded se Supabase indisponível ou seções todas vazias.

    Também propaga ai_model/ai_temperature/ai_max_tokens override pro _make_llm.
    """
    from panel.cache import compose_system_prompt, get_project_config_cache

    project_id = state.get("project_id") or os.getenv("DEFAULT_PROJECT_ID", "padrao")
    cfg = await get_project_config_cache().get(project_id)

    # Compõe prompt das seções; se nenhum conteúdo, fallback hardcoded.
    composed = compose_system_prompt(cfg)
    system_prompt = composed or SALES_SYSTEM

    patch: dict[str, Any] = {"system_prompt": system_prompt}
    if cfg.get("ai_model"):
        patch["_ai_model"] = cfg["ai_model"]
    if cfg.get("ai_temperature") is not None:
        try:
            patch["_ai_temperature"] = float(cfg["ai_temperature"])
        except (TypeError, ValueError):
            pass
    if cfg.get("ai_max_tokens"):
        try:
            patch["_ai_max_tokens"] = int(cfg["ai_max_tokens"])
        except (TypeError, ValueError):
            pass
    return patch


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
                    else:
                        log.warning("[flow] sendMedia falhou %s: %s", r.status_code, r.text[:160])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("[flow] step %s falhou: %s", step.get("type"), exc)
            # Não derruba o grafo — segue pro próximo step.
            continue

    return {"flow_dispatched": True, "sent": sent > 0, "sent_count": sent}


# ────────────────────────────────────────────────────────────────────
# Nó: tenant_resolver (project_id via Supabase instance_projects)
# ────────────────────────────────────────────────────────────────────

async def supervisor_node(state: SalesState) -> dict[str, Any]:
    """
    Valida a resposta proposta pelo especialista ANTES de persist+send.

    Roda APÓS especialistas (greeting/respond/close/etc) e ANTES de strategist.
    Se rejected E attempts < max, NÃO bloqueia mas marca pra retry no especialista.

    NOTA: retry real é feito via re-routing condicional no graph. Aqui só decide.
    """
    if os.getenv("SUPERVISOR_DISABLED") == "1":
        return {}
    # Skip se não há reply (intent=comprou path, follow-up sem msg, etc)
    proposed = state.get("reply") or ""
    if not proposed.strip():
        return {}

    from agent.supervisor import review_reply, SUPERVISOR_MAX_RETRIES

    attempts = state.get("supervisor_attempts", 0) or 0

    try:
        review = await review_reply(
            proposed_reply=proposed,
            messages=state.get("messages") or [],
            lead_facts=state.get("lead_facts"),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[supervisor] erro: %s — default approve", exc)
        review = {"approved": True, "reason": "supervisor error", "feedback": None, "severity": "ok"}

    log.info(
        "[supervisor] %s/%s approved=%s severity=%s reason=%s",
        state.get("instance_name", "?"), state.get("phone", "?"),
        review["approved"], review["severity"], review["reason"],
    )

    patch: dict[str, Any] = {"supervisor_review": review}

    # Se rejeitado E ainda tem retry disponível, marca pra refazer
    if not review["approved"] and attempts < SUPERVISOR_MAX_RETRIES:
        patch["supervisor_feedback"] = review.get("feedback") or review.get("reason") or ""
        patch["supervisor_attempts"] = attempts + 1
        # Limpa chunks/reply pra forçar especialista regerar
        patch["reply"] = ""
        patch["chunks"] = []
    else:
        # Approved OU esgotou retries — segue pra strategist.
        # CRÍTICO: limpa supervisor_feedback/attempts pra não vazar pro próximo
        # turno (checkpointer persiste o state — feedback velho contaminaria o
        # respond_node do próximo invoke do graph).
        if not review["approved"]:
            log.warning(
                "[supervisor] %s/%s max retries (%d) atingido, envia mesmo. reason=%s",
                state.get("instance_name", "?"), state.get("phone", "?"),
                SUPERVISOR_MAX_RETRIES, review["reason"],
            )
        patch["supervisor_feedback"] = ""
        patch["supervisor_attempts"] = 0

    return patch


async def follow_up_strategist_node(state: SalesState) -> dict[str, Any]:
    """
    Roda DEPOIS dos especialistas (que geraram resposta) e ANTES de persist+send.

    Analisa contexto + classifica temperatura + decide cadência ótima.
    Sobrescreve state["schedule_minutes"] com decisão baseada em psicologia
    comportamental (vs. tag [AGENDAR: N] vaga emitida pelo LLM principal).

    Pula:
      - intent == "follow_up" (bot disparando follow-up — não decide próximo agora)
      - intent == "comprou"   (lead converteu — sem follow-up)
      - has_converted == True
      - STRATEGIST_DISABLED env (tests, dev)
    """
    if os.getenv("STRATEGIST_DISABLED") == "1":
        return {}
    intent = state.get("intent") or ""
    # Pula apenas em comprou — em follow_up roda também pra decidir próximo
    # intervalo com base em psicologia (não confiar só na tag [AGENDAR:N]
    # vinda do follow_up_node).
    if intent == "comprou" or state.get("has_converted"):
        return {}

    from agent.follow_up_strategist import classify_lead

    redis = get_redis()
    instance = state.get("instance_name", "")
    phone = state.get("phone", "")

    # Lê contador ANTES de classificar (hard cap depende disso).
    try:
        attempts = await redis.get_followup_attempts(instance, phone)
    except Exception:  # noqa: BLE001
        attempts = 0

    # Em follow_up turn, state.user_message está vazio (bot inicia).
    # Pega a última mensagem REAL do cliente do histórico pra contexto.
    last_user_msg = state.get("user_message", "") or ""
    if not last_user_msg:
        msgs = state.get("messages") or []
        for m in reversed(msgs):
            if getattr(m, "type", "") == "human":
                content = getattr(m, "content", "")
                if isinstance(content, str) and content.strip():
                    last_user_msg = content
                    break

    try:
        decision = await classify_lead(
            messages=state.get("messages") or [],
            last_user_message=last_user_msg,
            attempts_made=attempts,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[strategist] erro inesperado: %s — usando fallback WARM 90min", exc)
        decision = {
            "temperatura": "WARM",
            "razao": f"strategist erro: {exc!s}"[:120],
            "horario_explicito": None,
            "agendar_minutos": 90,
            "abordagem": "valor",
            "killswitch_permanent": False,
        }

    # Reset SÓ em engagement real: intent forte OU temperatura HOT/SCHEDULED.
    # Cliente respondendo "ok"/"valeu" (intent=outros) NÃO zera o counter.
    engagement_intents = {"intencao_compra", "duvida_produto", "pedir_preco", "objecao"}
    engagement_temps = {"HOT", "SCHEDULED"}
    if (
        intent in engagement_intents
        or decision.get("temperatura") in engagement_temps
    ):
        try:
            await redis.reset_followup_attempts(instance, phone)
            attempts = 0
            log.info("[strategist] reset attempts %s/%s (engagement: intent=%s temp=%s)",
                     instance, phone, intent, decision.get("temperatura"))
        except Exception:  # noqa: BLE001
            pass

    log.info(
        "[strategist] %s/%s temp=%s min=%d abord=%s razao=%s",
        instance, phone,
        decision["temperatura"],
        decision["agendar_minutos"],
        decision["abordagem"],
        decision["razao"],
    )

    # Grava snapshot pro painel Reconquista
    from datetime import datetime, timedelta, timezone
    now_utc = datetime.now(timezone.utc)
    next_followup_at = None
    if decision["agendar_minutos"] and decision["agendar_minutos"] > 0:
        next_followup_at = (
            now_utc + timedelta(minutes=decision["agendar_minutos"])
        ).isoformat()
    snapshot = {
        "project_id": state.get("project_id") or "padrao",
        "instance": instance,
        "phone": phone,
        "push_name": state.get("push_name", ""),
        "temperatura": decision["temperatura"],
        "razao": decision["razao"],
        "abordagem": decision["abordagem"],
        "agendar_minutos": decision["agendar_minutos"],
        "killswitch_permanent": decision["killswitch_permanent"],
        "attempts_made": attempts,
        "last_decision_at": now_utc.isoformat(),
        "next_followup_at": next_followup_at,
        "intent": intent,
    }
    try:
        await redis.set_lead_status(instance, phone, snapshot)
    except Exception:  # noqa: BLE001
        pass

    patch: dict[str, Any] = {
        "follow_up_strategy": decision,
        "follow_up_attempts": attempts,
    }
    # Killswitch ou STOP → não agenda
    if decision["killswitch_permanent"] or decision["agendar_minutos"] <= 0:
        patch["schedule_minutes"] = None
    else:
        patch["schedule_minutes"] = decision["agendar_minutos"]

    return patch


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

    evo = get_evolution()
    instance = state["instance_name"]
    phone = state["phone"]
    message_id = state.get("message_id", "")
    redis = get_redis()

    # Cooldowns server-side — bloqueia spam mesmo se LLM emitir tags em sequência
    REACT_COOLDOWN_S = 240   # 4 min
    QUOTE_COOLDOWN_S = 180   # 3 min

    # Marca mensagem do cliente como LIDA antes de qualquer resposta
    # → faz aparecer ✓✓ azul (2 tiques azuis) no WhatsApp do cliente
    # → sinaliza que bot recebeu + leu, antes do "digitando..."
    # → roda mesmo em caminhos silenciosos (chunks vazios, flow_dispatched)
    # Marca TODAS as msgs unread do buffer (rajada do cliente → ✓✓ azul em todas).
    # Drena Redis buffer + inclui current message_id pra cobertura completa.
    remote_jid = f"{phone}@s.whatsapp.net"
    unread_ids: list[str] = []
    try:
        unread_ids = await redis.drain_unread(instance, phone)
    except Exception:  # noqa: BLE001
        unread_ids = []
    if message_id and message_id not in unread_ids:
        unread_ids.append(message_id)
    if unread_ids:
        try:
            await evo.mark_messages_read(instance, remote_jid, unread_ids)
            log.info("[send] mark_read %s/%s ids=%d", instance, phone, len(unread_ids))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("[send] mark_read falhou (cosmético): %s", exc)

    if state.get("flow_dispatched"):
        return {}

    chunks = state.get("chunks") or []
    if not chunks:
        return {"sent": False, "sent_count": 0}

    # Reação opcional — cooldown server-side garante não-spam
    react_emoji = state.get("react_emoji")
    if react_emoji and message_id:
        try:
            last_react = await redis._cmd("GET", f"react_last:{instance}:{phone}")
            if not last_react:
                await evo.send_reaction(instance, phone, message_id, react_emoji)
                await redis._cmd("SET", f"react_last:{instance}:{phone}", "1",
                                 "EX", str(REACT_COOLDOWN_S))
                log.info("[send] react %s/%s emoji=%s", instance, phone, react_emoji)
            else:
                log.debug("[send] react cooldown ativo — pula %s/%s", instance, phone)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.debug("[send] reaction falhou (cosmético): %s", exc)

    # Quote — só na primeira bolha, com cooldown server-side.
    # Regra extra: SE cliente fez pergunta ("?"), força quote (override LLM).
    # Cooldown segue valendo — 2 perguntas em rajada → só 1ª recebe quote.
    quote_id: str | None = None
    quote_text: str | None = None
    user_msg = state.get("user_message", "") or ""
    is_question = "?" in user_msg
    should_quote = bool(state.get("quote_previous")) or is_question
    if should_quote and message_id:
        try:
            last_quote = await redis._cmd("GET", f"quote_last:{instance}:{phone}")
            if not last_quote:
                quote_id = message_id
                quote_text = user_msg or None
                await redis._cmd("SET", f"quote_last:{instance}:{phone}", "1",
                                 "EX", str(QUOTE_COOLDOWN_S))
                reason = "pergunta" if (is_question and not state.get("quote_previous")) else "llm"
                log.info("[send] quote %s/%s msg=%s reason=%s",
                         instance, phone, message_id, reason)
            else:
                log.debug("[send] quote cooldown ativo — pula %s/%s", instance, phone)
        except Exception as exc:  # noqa: BLE001
            log.debug("[send] quote cooldown check falhou: %s", exc)

    sent = 0
    for i, chunk in enumerate(chunks):
        delay = typing_delay_ms(chunk)
        # Quote só primeira bolha (chunks subsequentes = continuação)
        chunk_quote_id = quote_id if i == 0 else None
        chunk_quote_text = quote_text if i == 0 else None
        try:
            await evo.send_typing(instance, phone, duration_ms=delay)
            await asyncio.sleep(delay / 1000)
            r = await evo.send_text(
                instance, phone, chunk,
                quoted_msg_id=chunk_quote_id,
                quoted_msg_text=chunk_quote_text,
            )
            if r.get("success"):
                sent += 1
            if i < len(chunks) - 1:
                await asyncio.sleep(jitter_between_bubbles_ms() / 1000)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("[send] bolha %d falhou: %s", i, exc)
            # Não derruba o grafo — registra parcial e segue.
            continue

    return {"sent": sent > 0, "sent_count": sent}
