"""
Lead Memory — extração e persistência de fatos estruturados do lead.

Diferente do histórico de mensagens (RedisStore.chat_history) que é texto bruto,
aqui guardamos FATOS extraídos via LLM para o bot SABER coisas sem reler tudo:

  - plataforma do lead (PC/Console/celular/TV/xCloud/nuvem/misto)
  - plano que o lead se interessou (R$40 3m / R$80 privada / etc)
  - nome (se descobriu)
  - objeções já levantadas (caro/vou pensar/já tenho/etc)
  - estágio da conversa (saudacao/descoberta/apresentacao/preco/objecao/fechamento/pos_venda)
  - já recebeu link de pagamento?
  - explicitamente confirmou compra?

Injetado no system prompt como <lead_conhecido>. Bot NUNCA pergunta o que já sabe.

Storage: Redis key `lead_facts:{instance}:{phone}` TTL 30 dias.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)


LEAD_FACTS_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 dias
FACTS_MODEL = os.getenv("LEAD_FACTS_MODEL", "openai/gpt-4o-mini")
FACTS_TIMEOUT = float(os.getenv("LEAD_FACTS_TIMEOUT", "12"))


# Schema canônico — adicionar novo campo aqui + atualizar EXTRACT_PROMPT
FACTS_SCHEMA: dict[str, Any] = {
    "plataforma":          None,   # PC | Console | celular | TV | xCloud | nuvem | misto | None
    "nome":                None,   # string ou None
    "plano_interesse":     None,   # "compartilhada_3m_40" | "compartilhada_1m_20" | "privada_80" | "privada_60" | None
    "objecoes":            [],     # ["caro", "vou_pensar", "ja_tenho", "nao_confio", "depois", ...]
    "estagio":             "descoberta",  # descoberta | apresentacao | preco | objecao | fechamento | pos_venda
    "ja_recebeu_link":     False,
    "confirmou_compra":    False,
    "ultimo_resumo":       "",     # 1-frase resumo da conversa ate aqui (pra continuidade)
}


EXTRACT_PROMPT = """\
Você é um analista de vendas. Sua tarefa: extrair FATOS estruturados de uma conversa
de vendas WhatsApp e ATUALIZAR o estado conhecido do lead.

Lê o histórico completo abaixo e o estado atual. Retorne SOMENTE um JSON com o
estado ATUALIZADO (não delta). NUNCA invente; se não tem certeza, use null/false/[].

ESTADO DO LEAD — formato JSON exato:
{
  "plataforma":          "PC" | "Console" | "celular" | "TV" | "xCloud" | "nuvem" | "misto" | null,
  "nome":                "<nome se mencionado>" | null,
  "plano_interesse":     "compartilhada_3m_40" | "compartilhada_1m_20" | "compartilhada_15d_10" | "privada_80" | "privada_60" | null,
  "objecoes":            ["caro" | "vou_pensar" | "ja_tenho" | "nao_confio" | "depois" | "sem_dinheiro" | "tem_pirateria"],
  "estagio":             "descoberta" | "apresentacao" | "preco" | "objecao" | "fechamento" | "pos_venda",
  "ja_recebeu_link":     true | false,
  "confirmou_compra":    true | false,
  "ultimo_resumo":       "<= 80 chars resumindo o que aconteceu pra agente continuar do ponto certo>"
}

REGRAS:
- plataforma: se cliente disse "PC", "computador", "no PC", "joga no PC" → "PC". Se disse "Xbox", "PlayStation", "PS5", "console" → "Console". Etc.
- nome: só preencha se cliente literalmente disse o nome dele (ex: "sou o João"). NUNCA invente.
- plano_interesse: tracking de qual link/preço o agente OFERECEU ou o cliente PEDIU.
    compartilhada_3m_40 = R$40 por 3 meses (mais comum oferecido primeiro)
    privada_80 = R$80 mês conta privada (TV/celular/xCloud/nuvem)
- objecoes: lista cumulativa. Adicione novas se aparecerem, mantenha antigas.
- estagio: progressão natural. Não regride.
    descoberta = ainda nao sabe plataforma do lead
    apresentacao = plataforma sabida, mostrando produto
    preco = ofereceu valor, esperando reação
    objecao = lead hesitou
    fechamento = ofereceu link/pix
    pos_venda = comprou
- ultimo_resumo: 1 frase tipo "lead PC interessado em 3m, ainda nao decidiu" — pra agente saber onde retomar.

RETORNE SOMENTE O JSON. SEM markdown, sem ```, sem explicação.
"""


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=FACTS_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "missing",
        base_url=os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.1,
        max_tokens=400,
        timeout=FACTS_TIMEOUT,
        default_headers={
            "HTTP-Referer": os.getenv("AI_REFERRER", "https://bot-vendas.local"),
            "X-Title": "bot-vendas-lead-facts",
        },
    )


def _conversation_to_text(messages: list[BaseMessage], limit: int = 30) -> str:
    recent = messages[-limit:] if len(messages) > limit else messages
    lines: list[str] = []
    for m in recent:
        msg_type = getattr(m, "type", "")
        role = "AGENT" if msg_type == "ai" else ("CLIENTE" if msg_type == "human" else msg_type.upper())
        text = getattr(m, "content", "")
        if isinstance(text, str) and text.strip():
            lines.append(f"{role}: {text[:280]}")
    return "\n".join(lines) or "(conversa vazia)"


def empty_facts() -> dict[str, Any]:
    """Estado default — usado quando lead novo."""
    return {**FACTS_SCHEMA, "objecoes": []}


def _validate_facts(raw: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Sanitiza output LLM — preenche defaults faltantes, valida enums."""
    out = empty_facts()
    out.update(current)  # parte do current pra não regredir
    # Aplica novos campos válidos
    if isinstance(raw, dict):
        plataforma = raw.get("plataforma")
        if plataforma in ("PC", "Console", "celular", "TV", "xCloud", "nuvem", "misto"):
            out["plataforma"] = plataforma
        nome = raw.get("nome")
        if isinstance(nome, str) and 1 <= len(nome) <= 50:
            out["nome"] = nome
        plano = raw.get("plano_interesse")
        if plano in (
            "compartilhada_3m_40", "compartilhada_1m_20", "compartilhada_15d_10",
            "privada_80", "privada_60",
        ):
            out["plano_interesse"] = plano
        objecoes_raw = raw.get("objecoes") or []
        if isinstance(objecoes_raw, list):
            valid_obj = {"caro", "vou_pensar", "ja_tenho", "nao_confio", "depois", "sem_dinheiro", "tem_pirateria"}
            new_obj = list(set(out.get("objecoes", []) + [o for o in objecoes_raw if o in valid_obj]))
            out["objecoes"] = new_obj
        estagio = raw.get("estagio")
        if estagio in ("descoberta", "apresentacao", "preco", "objecao", "fechamento", "pos_venda"):
            # Não regride: pos_venda > fechamento > preco/objecao > apresentacao > descoberta
            ranking = {"descoberta": 0, "apresentacao": 1, "preco": 2, "objecao": 2, "fechamento": 3, "pos_venda": 4}
            cur_rank = ranking.get(out.get("estagio", "descoberta"), 0)
            new_rank = ranking.get(estagio, 0)
            if new_rank >= cur_rank:
                out["estagio"] = estagio
        if isinstance(raw.get("ja_recebeu_link"), bool):
            out["ja_recebeu_link"] = raw["ja_recebeu_link"] or out.get("ja_recebeu_link", False)
        if isinstance(raw.get("confirmou_compra"), bool):
            out["confirmou_compra"] = raw["confirmou_compra"] or out.get("confirmou_compra", False)
        ult = raw.get("ultimo_resumo")
        if isinstance(ult, str) and ult.strip():
            out["ultimo_resumo"] = ult.strip()[:160]
    return out


async def extract_facts(
    messages: list[BaseMessage],
    current_facts: dict[str, Any],
) -> dict[str, Any]:
    """
    Roda LLM extraction. Retorna facts atualizado (sempre dict válido).
    Fallback safe: se LLM falhar, devolve current_facts inalterado.
    """
    if not messages:
        return current_facts

    convo = _conversation_to_text(messages)
    user_prompt = (
        f"=== HISTÓRICO DA CONVERSA ===\n{convo}\n\n"
        f"=== ESTADO ATUAL DO LEAD ===\n{json.dumps(current_facts, ensure_ascii=False)}\n\n"
        "Retorne JSON atualizado conforme regras."
    )

    try:
        llm = _build_llm()
        result = await llm.ainvoke([
            SystemMessage(content=EXTRACT_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        raw = (result.content or "").strip()
        # Remove cercas markdown
        import re as _re
        raw = _re.sub(r"^```(?:json)?\s*", "", raw)
        raw = _re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("[lead_facts] parse falhou (%s) — mantém current", exc)
        return current_facts
    except Exception as exc:  # noqa: BLE001
        log.warning("[lead_facts] LLM erro (%s) — mantém current", exc)
        return current_facts

    return _validate_facts(parsed, current_facts)


def format_for_prompt(facts: dict[str, Any]) -> str:
    """
    Renderiza facts como bloco pro system prompt. Empty-safe.
    Output exemplo:
        <lead_conhecido>
        Plataforma: PC (confirmada)
        Nome: João
        Plano discutido: 3 meses compartilhada R$40
        Objeções já tratadas: caro
        Estágio atual: preço (esperando reação do lead)
        Já recebeu link: não
        Última situação: lead achou caro, agente vai reforçar valor.
        </lead_conhecido>

        REGRA CRÍTICA: NUNCA pergunte de novo o que já está acima.
        Avance no estágio. NÃO recomece pitch nem peça plataforma se já souber.
    """
    if not facts:
        return ""

    lines: list[str] = []
    if facts.get("plataforma"):
        lines.append(f"Plataforma do lead: {facts['plataforma']} (confirmada)")
    if facts.get("nome"):
        lines.append(f"Nome: {facts['nome']}")
    if facts.get("plano_interesse"):
        plano_map = {
            "compartilhada_3m_40":  "3 meses compartilhada R$40",
            "compartilhada_1m_20":  "1 mês compartilhada R$20",
            "compartilhada_15d_10": "15 dias compartilhada R$10",
            "privada_80":           "1 mês conta privada R$80",
            "privada_60":           "1 mês conta privada R$60",
        }
        lines.append(f"Plano discutido: {plano_map.get(facts['plano_interesse'], facts['plano_interesse'])}")
    if facts.get("objecoes"):
        lines.append(f"Objeções já tratadas: {', '.join(facts['objecoes'])}")
    estagio = facts.get("estagio") or "descoberta"
    estagio_descr = {
        "descoberta":   "descoberta (ainda não sabe plataforma)",
        "apresentacao": "apresentação (sabe plataforma, mostrando produto)",
        "preco":        "preço (ofereceu valor, esperando reação)",
        "objecao":      "objeção (lead hesitou, contornando)",
        "fechamento":   "fechamento (ofereceu link/pix, esperando pagamento)",
        "pos_venda":    "pós-venda (cliente comprou)",
    }
    lines.append(f"Estágio atual: {estagio_descr.get(estagio, estagio)}")
    if facts.get("ja_recebeu_link"):
        lines.append("Já recebeu link de pagamento: SIM")
    if facts.get("confirmou_compra"):
        lines.append("Cliente JÁ COMPROU. Modo pós-venda.")
    if facts.get("ultimo_resumo"):
        lines.append(f"Última situação: {facts['ultimo_resumo']}")

    if not lines:
        return ""

    return (
        "<lead_conhecido>\n"
        + "\n".join(lines)
        + "\n</lead_conhecido>\n\n"
        + "REGRAS CRÍTICAS:\n"
        + "1. NUNCA pergunte de novo o que está acima (plataforma, nome, plano).\n"
        + "2. NÃO recomece o pitch nem se apresente de novo.\n"
        + "3. AVANCE no estágio: depois de descoberta vem apresentação, depois preço, depois fechamento.\n"
        + "4. Se cliente parece confuso ('como assim?', '?'), peça desculpa CURTA + continue de onde parou.\n"
    )
