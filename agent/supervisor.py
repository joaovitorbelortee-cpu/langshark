"""
Supervisor Agent — valida resposta do especialista ANTES de enviar.

Problema identificado: especialistas (greeting/respond/close/objection/follow_up)
às vezes violam regras do prompt:
  - Mandar link de pagamento sem cliente confirmar interesse
  - Expor planos secretos (R$10 dificuldade financeira / R$60 após 3 objeções)
  - Pular etapas (recomendar antes de descobrir plataforma)
  - Repetir info já dita
  - Spam de perguntas

Supervisor = LLM independente que SO analisa: "essa resposta segue as regras?
faz sentido no contexto? cliente pediu isso?". Retorna {approved, reason, feedback}.
Se rejected, especialista refaz com feedback (max 2 retries).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "openai/gpt-4o-mini")
SUPERVISOR_TIMEOUT = float(os.getenv("SUPERVISOR_TIMEOUT", "20"))
SUPERVISOR_MAX_RETRIES = int(os.getenv("SUPERVISOR_MAX_RETRIES", "2"))
SUPERVISOR_DISABLED = os.getenv("SUPERVISOR_DISABLED") == "1"


SUPERVISOR_PROMPT = """\
Você é o SUPERVISOR DE VENDAS. Sua tarefa: analisar se a resposta proposta
pelo agente de vendas segue as regras estabelecidas no system prompt principal
E faz sentido no contexto da conversa.

VOCÊ NÃO ESCREVE A RESPOSTA. VOCÊ APENAS APROVA OU REJEITA.

═══ CHECAGEM CRÍTICA ═══

1. PLATAFORMA-FIRST: Agente perguntou a plataforma do lead ANTES de oferecer
   plano? Se cliente nunca disse PC/Console/celular/TV, agente NÃO PODE recomendar
   um plano. Plataforma indefinida → APENAS pergunta.

2. LINK DE PAGAMENTO: Agente só pode mandar link quando:
   a) Cliente CONFIRMOU interesse no plano específico ("vou pegar", "pode mandar
      o link", "quero esse"), OU
   b) Cliente PEDIU o link explicitamente.
   Se agente mandou link SEM cliente confirmar → REJEITAR.

3. PLANOS RESTRITOS (NÃO PODE EXPOR ESPONTANEAMENTE):
   - R$10 (15 dias compartilhada): SOMENTE quando cliente disser "tô sem dinheiro",
     "não tenho como pagar agora", "tá apertado financeiro". Caso contrário NUNCA.
   - R$60 (privada com desconto): SOMENTE após 3+ objeções reais de preço.
     "tá caro" 1x não conta — precisa ser 3 vezes ou negociação clara.
   Se agente listou esses planos em resposta normal a "quais os planos?" → REJEITAR.

4. ANTI-REPETIÇÃO: Resposta repete pergunta já feita no histórico? (ex: "qual
   plataforma você joga?" quando plataforma já mencionada).

5. ESTÁGIO COERENTE: Resposta combina com estágio da conversa em <lead_conhecido>?
   - Estágio=descoberta → agente DEVE perguntar plataforma se ainda não sabe
   - Estágio=apresentacao → agente mostra produto sem mandar link
   - Estágio=preco → cliente sabe valor, esperando reação. Agente espera, não força
   - Estágio=fechamento → cliente quer comprar. Agente pode mandar link agora.

6. CALIDADE HUMANA:
   - Sem frases batidas tipo "Como posso te ajudar?", "Sou seu vendedor...".
   - Tom natural, casual no nível certo (igual ao lead).
   - Sem invencionices (preço, prazo, garantia que não existe).
   - PONTUAÇÃO: rejeite respostas que terminam TODA frase com "." ou "!".
     Soa robótico. Humanos no WhatsApp variam: às vezes sem ponto, "...",
     vírgula, só quebra. "!" só pra emoção REAL.
     EX RUIM: "Beleza! Qual plataforma você joga?"
     EX BOM:  "beleza, qual plataforma vc joga"

7. AÇÕES PRECIPITADAS:
   - "te mando o link em N minutos" mas cliente NÃO pediu = ruim.
   - Pular etapas (oferecer plano sem descobrir contexto).

═══ FORMATO DE RESPOSTA ═══

Retorne SOMENTE JSON puro, sem markdown:
{
  "approved": true | false,
  "reason": "<descrição curta < 60 chars do diagnóstico>",
  "feedback": "<instrução EM 1 FRASE pro agente refazer, ou null se approved>",
  "severity": "ok" | "warning" | "critical"
}

EXEMPLOS:

Caso A — Aprovado:
{
  "approved": true,
  "reason": "resposta coerente com estágio",
  "feedback": null,
  "severity": "ok"
}

Caso B — Rejeitado (mandou link sem confirmação):
{
  "approved": false,
  "reason": "mandou link sem cliente pedir/confirmar plano",
  "feedback": "Cliente perguntou outros planos mas não confirmou que vai pegar. Liste os 2 principais (R$40 3m e R$20 1m), pergunta qual interessou. NÃO mande link agora.",
  "severity": "critical"
}

Caso C — Rejeitado (expôs R$10 sem trigger):
{
  "approved": false,
  "reason": "expôs R$10/15dias sem trigger de dificuldade financeira",
  "feedback": "R$10 é só pra quem disser sem dinheiro. Liste APENAS R$40 (3m compartilhada) e R$80 (privada). Pergunta qual.",
  "severity": "critical"
}

Caso D — Rejeitado (resposta robótica):
{
  "approved": false,
  "reason": "linguagem genérica de bot",
  "feedback": "Reescreve mais casual. Sem 'Como posso te ajudar?'. Cliente disse 'oi' + tem histórico — retoma do estágio.",
  "severity": "warning"
}
"""


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=SUPERVISOR_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "missing",
        base_url=os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.1,
        max_tokens=300,
        timeout=SUPERVISOR_TIMEOUT,
        default_headers={
            "HTTP-Referer": os.getenv("AI_REFERRER", "https://bot-vendas.local"),
            "X-Title": "bot-vendas-supervisor",
        },
    )


def _conversation_to_text(messages: list[BaseMessage], limit: int = 20) -> str:
    recent = messages[-limit:] if len(messages) > limit else messages
    lines: list[str] = []
    for m in recent:
        msg_type = getattr(m, "type", "")
        role = "AGENT" if msg_type == "ai" else ("CLIENTE" if msg_type == "human" else msg_type.upper())
        text = getattr(m, "content", "")
        if isinstance(text, str) and text.strip():
            lines.append(f"{role}: {text[:280]}")
    return "\n".join(lines) or "(conversa vazia)"


async def review_reply(
    proposed_reply: str,
    messages: list[BaseMessage],
    lead_facts: dict[str, Any] | None,
    system_rules_summary: str = "",
) -> dict[str, Any]:
    """
    Pede pro supervisor LLM avaliar a resposta proposta.

    Sempre retorna dict válido. Em caso de erro, default = approved (não bloqueia
    bot por falha de supervisor).
    """
    if SUPERVISOR_DISABLED:
        return {"approved": True, "reason": "supervisor disabled", "feedback": None, "severity": "ok"}

    if not proposed_reply or not proposed_reply.strip():
        return {"approved": False, "reason": "reply vazia", "feedback": "Escreva uma resposta.", "severity": "warning"}

    convo = _conversation_to_text(messages)
    facts_str = json.dumps(lead_facts or {}, ensure_ascii=False, indent=2)

    user_prompt = (
        f"═══ HISTÓRICO DA CONVERSA ═══\n{convo}\n\n"
        f"═══ ESTADO CONHECIDO DO LEAD ═══\n{facts_str}\n\n"
        f"═══ RESPOSTA PROPOSTA PELO AGENTE ═══\n{proposed_reply}\n\n"
        "Avalie e retorne JSON conforme regras do system prompt."
    )

    try:
        llm = _build_llm()
        result = await llm.ainvoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        raw = (result.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("[supervisor] parse falhou (%s) — default approve", exc)
        return {"approved": True, "reason": "supervisor parse error", "feedback": None, "severity": "ok"}
    except Exception as exc:  # noqa: BLE001
        log.warning("[supervisor] LLM erro (%s) — default approve", exc)
        return {"approved": True, "reason": "supervisor llm error", "feedback": None, "severity": "ok"}

    # Sanitize
    approved = bool(parsed.get("approved", True))
    reason = str(parsed.get("reason") or "")[:120]
    feedback = parsed.get("feedback") if not approved else None
    if feedback and isinstance(feedback, str):
        feedback = feedback[:300]
    severity = str(parsed.get("severity") or "ok").lower()
    if severity not in ("ok", "warning", "critical"):
        severity = "warning"

    return {
        "approved": approved,
        "reason": reason,
        "feedback": feedback,
        "severity": severity,
    }


def format_feedback_for_retry(feedback: str | None, reason: str | None = None) -> str:
    """Renderiza feedback do supervisor pra injetar no system prompt do retry."""
    if not feedback and not reason:
        return ""
    lines = ["<supervisor_feedback>"]
    if reason:
        lines.append(f"PROBLEMA NA RESPOSTA ANTERIOR: {reason}")
    if feedback:
        lines.append(f"INSTRUÇÃO PRO RETRY: {feedback}")
    lines.append("Reescreva a resposta seguindo essa correção. NÃO repita o erro.")
    lines.append("</supervisor_feedback>")
    return "\n".join(lines)
