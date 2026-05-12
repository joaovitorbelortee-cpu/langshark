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


# ──────────────────────────────────────────────────────────────────────
# Anti-repetição programática (fail-fast SEM LLM)
# ──────────────────────────────────────────────────────────────────────

def _normalize_for_compare(text: str) -> str:
    """Lowercase + remove pontuação/espaços extras pra comparar conteúdo."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\sáéíóúâêîôûãõçà]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _word_set(text: str) -> set[str]:
    """Tokens >= 3 chars (evita stopwords pegando 1-2 char)."""
    return {w for w in _normalize_for_compare(text).split() if len(w) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _check_repetition(proposed: str, messages: list[BaseMessage], lookback: int = 4) -> dict[str, Any] | None:
    """
    Compara `proposed` com últimas N AIMessages do histórico. Se overlap alto =
    repetição → reject.

    CRITICAL: pula a PRIMEIRA AIMessage encontrada (de trás pra frente). Quando
    supervisor roda, o especialista já appended o proposed no state.messages.
    Não comparar com ele mesmo (self-match 100% causa reject infinito).

    Thresholds:
      - jaccard >= 0.55 (>55% palavras em comum) → CRITICAL
      - jaccard >= 0.40 → WARNING
    """
    if not proposed or not isinstance(proposed, str):
        return None
    new_set = _word_set(proposed)
    if len(new_set) < 4:
        return None  # frase muito curta, comparação não confiável

    proposed_norm = _normalize_for_compare(proposed)

    # Últimas N AIMessages — PULA a primeira encontrada se for igual ao proposed
    # (LLM já appended a msg no state antes do supervisor rodar)
    ai_msgs: list[str] = []
    skipped_self = False
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "ai":
            content = getattr(m, "content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            # Pula auto-match (primeiro AIMessage idêntico ao proposed)
            if not skipped_self and _normalize_for_compare(content) == proposed_norm:
                skipped_self = True
                continue
            ai_msgs.append(content)
            if len(ai_msgs) >= lookback:
                break
    if not ai_msgs:
        return None

    max_score = 0.0
    matched_text = ""
    for prev in ai_msgs:
        score = _jaccard(new_set, _word_set(prev))
        if score > max_score:
            max_score = score
            matched_text = prev

    if max_score >= 0.55:
        return {
            "approved": False,
            "reason": f"repetição forte ({max_score:.0%}) de msg anterior",
            "feedback": (
                "Você está REPETINDO uma frase quase idêntica que já mandou. "
                "Mude completamente as palavras E o ângulo. Cliente já leu o ponto antes; "
                "se for insistir, AVANCE no estágio (ex: dê alternativa, mude abordagem) "
                "ou pare de pedir a mesma coisa. Resposta anterior similar: "
                f"\"{matched_text[:120]}\""
            ),
            "severity": "critical",
        }
    if max_score >= 0.40:
        return {
            "approved": False,
            "reason": f"eco moderado ({max_score:.0%}) de msg anterior",
            "feedback": (
                "Sua resposta tem palavras demais em comum com uma anterior. "
                "Reescreva mudando estrutura E vocabulário. Não repita o mesmo pedido "
                "do mesmo jeito — varie completamente."
            ),
            "severity": "warning",
        }
    return None


_PRICE_PATTERNS = [
    re.compile(r"r\$\s*\d{1,3}", re.IGNORECASE),         # R$40, R$ 80
    re.compile(r"\bcompartilhad[ao]\b", re.IGNORECASE),  # plano compartilhado
    re.compile(r"\bprivad[ao]\b", re.IGNORECASE),        # plano privado
    re.compile(r"\d+\s*(meses|mês|mes)\b", re.IGNORECASE),  # 3 meses
    re.compile(r"\bplano de\s*\d", re.IGNORECASE),       # "plano de 40"
    re.compile(r"\b\d{2,3}\s*reais\b", re.IGNORECASE),   # "40 reais", "80 reais"
    re.compile(r"\bquarenta\s+reais\b", re.IGNORECASE),  # "quarenta reais" extenso
    re.compile(r"\boitenta\s+reais\b", re.IGNORECASE),
    re.compile(r"\bdez\s+reais\b", re.IGNORECASE),
    re.compile(r"\bsessenta\s+reais\b", re.IGNORECASE),
    re.compile(r"\bvinte\s+reais\b", re.IGNORECASE),
]

_PAYMENT_LINK_PATTERN = re.compile(
    r"(ggcheckout\.com|pagseguro\.com|mercadopago\.com|stripe\.com|"
    r"hotmart\.com|monetizze\.com|checkout/v2|"
    r"https?://[a-zA-Z0-9.-]+/(checkout|pagamento|pay|buy)/)",
    re.IGNORECASE,
)

# Keywords lead deve falar pra autorizar link
_CONFIRM_KEYWORDS = (
    "vou pegar", "vou querer", "quero esse", "quero pegar", "fechado",
    "manda o link", "manda link", "manda o pix", "manda pix", "pode mandar",
    "vou comprar", "to dentro", "fechou", "vamos fechar", "bora", "topo",
    "ja vou pagar", "vou pagar agora", "qual o pix", "qual link", "qual o link",
    "como pago", "onde pago",
)


def _check_link_without_confirmation(
    proposed: str,
    messages: list[BaseMessage],
    lead_facts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Bloqueia link de pagamento sem o lead ter CONFIRMADO interesse.

    Lead falar plataforma ("pc", "console") NÃO é confirmação de compra.
    Precisa frase explícita ("vou pegar", "manda o link", "quero fechar").
    """
    if not proposed:
        return None
    if not _PAYMENT_LINK_PATTERN.search(proposed):
        return None

    facts = lead_facts or {}
    # Lead já confirmou compra antes → libera
    if facts.get("confirmou_compra") or facts.get("ja_recebeu_link"):
        return None

    # Check últimas 3 msgs do user — alguma confirma intenção de compra?
    recent_user: list[str] = []
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "human":
            content = getattr(m, "content", "")
            if isinstance(content, str):
                recent_user.append(content.lower())
                if len(recent_user) >= 3:
                    break

    confirmed = any(
        kw in msg for msg in recent_user for kw in _CONFIRM_KEYWORDS
    )
    if confirmed:
        return None

    last_msg = recent_user[0] if recent_user else ""
    return {
        "approved": False,
        "reason": "mandou link sem cliente confirmar interesse",
        "feedback": (
            "Você mandou LINK DE PAGAMENTO mas o cliente NÃO confirmou que quer "
            "comprar. Ele falou apenas plataforma/dúvida básica, não 'vou pegar' "
            "ou 'manda o link'. REMOVA o link. Apresente o plano certo, mostra "
            "valor + vantagem, e pergunta tipo 'curtiu? te mando o link?' — "
            f"última msg do lead foi: \"{last_msg[:80]}\"."
        ),
        "severity": "critical",
    }


# Pontuação robotic patterns
_EXCLAM_LIMIT = 1  # max "!" por reply (stricter — 2+ "!" reject)
_PERIOD_AT_END_LIMIT = 0.6  # >60% frases terminando em "." = robotic

# Frases batidas que SOAM ROBOT — não usar
_CANNED_PHRASES = (
    "como posso te ajudar",
    "como posso ajudar",
    "estou aqui pra ajudar",
    "to aqui pra ajudar",
    "tô aqui pra ajudar",
    "estou à disposição",
    "estou a disposição",
    "qualquer dúvida estou",
    "qualquer duvida estou",
    "fique à vontade",
    "fique a vontade",
    "sou seu assistente",
    "sou um vendedor",
    "sou o vendedor",
    "como vai você",
    "espero que esteja bem",
    "tudo bem por aí",
    "se precisar de ajuda",  # ← spam follow-up clássico
    "se tiver alguma pergunta",
    "se tiver duvida",
    "se tiver dúvida",
    "pode contar comigo",
    "no que posso ajudar",
)


def _check_canned_phrases(proposed: str) -> dict[str, Any] | None:
    """Detecta frases batidas/template típicas de bot."""
    if not proposed:
        return None
    norm = proposed.lower()
    matched = next((p for p in _CANNED_PHRASES if p in norm), None)
    if matched:
        return {
            "approved": False,
            "reason": f"frase batida: \"{matched}\"",
            "feedback": (
                f"Você usou \"{matched}\" — frase batida de bot. NÃO use frases prontas. "
                "Escreva algo ESPECÍFICO ao contexto do lead. Se está em silêncio, "
                "puxa retomada baseada no ULTIMO_RESUMO do lead_conhecido (algo concreto "
                "da conversa anterior), não pergunta genérica. Sempre VARIE."
            ),
            "severity": "critical",
        }
    return None


def _check_punctuation(proposed: str) -> dict[str, Any] | None:
    """
    Detecta pontuação robotic: muitos "!" OU pontos finais em quase TODA frase.
    """
    if not proposed or len(proposed) < 20:
        return None
    # Conta "!" — 2+ exclamações = robot
    exclam = proposed.count("!")
    if exclam >= 2:
        return {
            "approved": False,
            "reason": f"{exclam} exclamações — robotic",
            "feedback": (
                f"Você usou {exclam} '!' na resposta. Soa entusiasmado-forçado de bot. "
                "REDUZ pra ZERO ou MÁXIMO 1. Use '!' SÓ em emoção real (lead fechou compra). "
                "Resto: nada, vírgula, ou reticências. Humano WhatsApp raramente usa '!'."
            ),
            "severity": "warning",
        }
    # Conta sentences terminando em "." (ignora ellipsis "..." que é casual humano)
    sentences = [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", proposed) if s.strip()]
    if len(sentences) >= 3:
        dot_endings = sum(
            1 for s in sentences
            if s.endswith(".") and not s.endswith("...")  # exclui reticências
        )
        ratio = dot_endings / len(sentences)
        if ratio >= _PERIOD_AT_END_LIMIT:
            return {
                "approved": False,
                "reason": f"{int(ratio*100)}% frases com '.' no fim — robotic",
                "feedback": (
                    "Quase TODA frase sua termina com '.'. Soa robot/escrito formal. "
                    "WhatsApp real ninguém pontua tudo certinho. Mistura: tira o '.' "
                    "da maioria, deixa só vírgula ou nada. Ex: 'beleza, vou olhar aqui' "
                    "em vez de 'Beleza. Vou olhar aqui.'"
                ),
                "severity": "warning",
            }
    return None


# Limites de comprimento por mensagem (WhatsApp humano = curto)
_MAX_TOTAL_CHARS = 600       # reply inteiro acima disso = mensagem-livro
_MAX_SINGLE_SENTENCE = 180   # frase única > 180 chars = compridona
_MAX_SENTENCES_NO_BREAK = 3  # >3 frases sem quebra de linha = parágrafo emendado


def _check_length(proposed: str) -> dict[str, Any] | None:
    """
    Detecta mensagem comprida/emendada (anti-humanização):
      - total > 600 chars
      - alguma frase única > 180 chars
      - 3+ frases sem quebra de linha (emendado em parágrafo)

    Bot WhatsApp deve enviar 1-3 bolhas CURTAS. Sistema chunk_for_whatsapp
    já fragmenta, mas se reply chegar gigantesco, fica parágrafo monstro.
    """
    if not proposed or len(proposed) < 50:
        return None

    total = len(proposed)
    if total > _MAX_TOTAL_CHARS:
        return {
            "approved": False,
            "reason": f"resposta gigantesca ({total} chars)",
            "feedback": (
                f"Sua resposta tem {total} caracteres. WhatsApp real ninguém manda "
                "texto-livro. CORTE: mantém o ESSENCIAL em 1-3 frases curtas. "
                "Termina com pergunta pra cliente responder. Resto deixa pro próximo turn."
            ),
            "severity": "warning",
        }

    # Sentences split (não corta abreviações)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", proposed) if s.strip()]
    for s in sentences:
        if len(s) > _MAX_SINGLE_SENTENCE:
            return {
                "approved": False,
                "reason": f"frase única gigante ({len(s)} chars)",
                "feedback": (
                    f"Você fez uma frase de {len(s)} chars sem cortar. Humano quebra "
                    "em mais de uma. PARTA em 2-3 frases curtas. Ex em vez de "
                    "'Olha, pra você que joga no PC ou Console o melhor plano que cobre "
                    "tudo seria a compartilhada de 3 meses' use 'pra PC/Console rola "
                    "compartilhada. R$40 por 3 meses. quer ver mais detalhe?'"
                ),
                "severity": "warning",
            }

    # Parágrafo emendado: várias frases SEM \n no meio
    if len(sentences) > _MAX_SENTENCES_NO_BREAK and "\n" not in proposed:
        return {
            "approved": False,
            "reason": f"{len(sentences)} frases emendadas sem quebra",
            "feedback": (
                f"Você emendou {len(sentences)} frases num parágrafo só. WhatsApp humano "
                "QUEBRA em msgs separadas. Use \\n pra dividir, OU corte pra 2-3 "
                "ideias maximo. Lead lê melhor msgs curtas."
            ),
            "severity": "warning",
        }
    return None


def _check_platform_first(
    proposed: str,
    lead_facts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Anti PLATAFORMA-FIRST: bot oferece preço/plano sem saber a plataforma do lead.

    Trigger: reply menciona valores/planos AND lead_facts.plataforma é falsy
    AND estagio == descoberta (ainda não descobriu plataforma).

    Retorna review reject; None se OK.
    """
    if not proposed:
        return None
    facts = lead_facts or {}
    plataforma = facts.get("plataforma")
    estagio = (facts.get("estagio") or "").lower()

    # Lead já tem plataforma conhecida → pode falar preço sem problema
    if plataforma:
        return None

    # Bot já avançou pra fechamento (lead deve ter dado contexto) — confia
    if estagio in ("fechamento", "pos_venda"):
        return None

    # Verifica se reply menciona preço/plano
    matched = None
    for pat in _PRICE_PATTERNS:
        m = pat.search(proposed)
        if m:
            matched = m.group(0)
            break
    if not matched:
        return None

    return {
        "approved": False,
        "reason": f"ofereceu preço/plano ('{matched}') sem saber plataforma",
        "feedback": (
            "Você mencionou preço/plano antes de descobrir a plataforma do lead. "
            "REGRA: PERGUNTE ANTES qual plataforma (PC/Console/celular/TV/xCloud) "
            "ele joga. SÓ DEPOIS, baseado na plataforma, ofereça o plano certo:\n"
            "  - PC ou Console → compartilhada (R$40 3 meses)\n"
            "  - Celular/TV/xCloud/nuvem → privada (R$80)\n"
            "REESCREVA: pergunta plataforma em tom casual, SEM listar planos ainda."
        ),
        "severity": "critical",
    }


def _check_compriou_fraud(
    proposed: str,
    messages: list[BaseMessage],
    lead_facts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Anti-fraude: bot vai mandar [COMPROU] mas não tem evidência de comprovante real?

    Triggers reject:
      - Reply tem [COMPROU] MAS lead nunca confirmou pagamento textualmente
      - Lead estagio != fechamento (lead nem chegou nessa etapa)
      - Última HumanMessage não foi imagem nem texto de confirmação ("paguei", "pix feito")

    Retorna review reject se suspeita; None se OK.
    """
    if not proposed or "[COMPROU]" not in proposed.upper():
        return None

    facts = lead_facts or {}
    estagio = (facts.get("estagio") or "").lower()

    # Lead JÁ confirmou compra anteriormente? Aceita
    if facts.get("confirmou_compra"):
        return None

    # Estágio precisa ser fechamento OU pos_venda (final do funnel)
    if estagio not in ("fechamento", "pos_venda"):
        return {
            "approved": False,
            "reason": f"[COMPROU] sem estágio fechamento (atual: {estagio or 'descoberta'})",
            "feedback": (
                "Você marcou [COMPROU] mas o lead nem chegou no estágio de fechamento. "
                "REMOVA a tag [COMPROU]. Não libere produto. "
                "Peça pra ele confirmar o pagamento antes — comprovante VÁLIDO com valor, status concluído e data."
            ),
            "severity": "critical",
        }

    # Checa APENAS as ÚLTIMAS 2 msgs do cliente — evidência precisa ser RECENTE.
    # Imagem antiga (3+ turnos atrás) NÃO conta como prova de pagamento atual.
    keywords_payment = (
        "paguei", "pago", "pix feito", "transferi", "comprovante",
        "pagamento feito", "ja paguei", "acabei de pagar", "fiz o pix",
        "boleto pago", "feito o pagamento",
    )
    last_user_msgs: list[str] = []
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "human":
            content = getattr(m, "content", "")
            if isinstance(content, str) and content.strip():
                last_user_msgs.append(content.lower())
                if len(last_user_msgs) >= 2:
                    break
            elif isinstance(content, list):
                # Multimodal (imagem) — é evidência de comprovante potencial
                last_user_msgs.append("[imagem]")
                if len(last_user_msgs) >= 2:
                    break

    has_payment_signal = any(
        kw in msg for msg in last_user_msgs for kw in keywords_payment
    )
    has_image_signal = any("[imagem]" in msg for msg in last_user_msgs)

    if not has_payment_signal and not has_image_signal:
        return {
            "approved": False,
            "reason": "[COMPROU] sem evidência (lead não disse 'paguei' nem mandou imagem)",
            "feedback": (
                "Você marcou [COMPROU] mas o lead NUNCA mandou comprovante nem disse que pagou. "
                "REMOVA [COMPROU]. Peça o comprovante (imagem com valor, status concluído, data) "
                "antes de liberar acesso."
            ),
            "severity": "critical",
        }

    return None


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

    # Fail-fast: repetição programática (sem custo LLM)
    rep_review = _check_repetition(proposed_reply, messages or [])
    if rep_review:
        log.info("[supervisor] anti-repetição rejeitou: %s", rep_review["reason"])
        return rep_review

    # Fail-fast: tag [COMPROU] sem evidência de comprovante real
    fraud_review = _check_compriou_fraud(proposed_reply, messages or [], lead_facts)
    if fraud_review:
        log.warning("[supervisor] anti-fraude COMPROU rejeitou: %s", fraud_review["reason"])
        return fraud_review

    # Fail-fast: bot oferecendo PREÇO sem saber plataforma do lead (PLATAFORMA-FIRST)
    platform_review = _check_platform_first(proposed_reply, lead_facts)
    if platform_review:
        log.warning("[supervisor] PLATAFORMA-FIRST rejeitou: %s", platform_review["reason"])
        return platform_review

    # Fail-fast: link de pagamento sem cliente confirmar interesse
    link_review = _check_link_without_confirmation(proposed_reply, messages or [], lead_facts)
    if link_review:
        log.warning("[supervisor] LINK sem confirmação rejeitou: %s", link_review["reason"])
        return link_review

    # Fail-fast: pontuação robotic (muitos ! ou . em toda frase)
    punct_review = _check_punctuation(proposed_reply)
    if punct_review:
        log.info("[supervisor] pontuação rejeitou: %s", punct_review["reason"])
        return punct_review

    # Fail-fast: frases batidas tipo "como posso te ajudar"
    canned_review = _check_canned_phrases(proposed_reply)
    if canned_review:
        log.info("[supervisor] frase batida rejeitou: %s", canned_review["reason"])
        return canned_review

    # Fail-fast: mensagem comprida/emendada (anti-humano)
    length_review = _check_length(proposed_reply)
    if length_review:
        log.info("[supervisor] comprimento rejeitou: %s", length_review["reason"])
        return length_review

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
        log.warning("[supervisor] parse falhou (%s) — default approve (passou 6 camadas programáticas)", exc)
        # Default approve só faz sentido AQUI porque já passou pelas 6 camadas
        # programáticas (anti-rep, fraud, plataforma, link, pontuação, length).
        # Se chegou aqui, reply já é razoável — só LLM contextual falhou parse.
        return {"approved": True, "reason": "supervisor parse error (camadas programáticas OK)", "feedback": None, "severity": "ok"}
    except Exception as exc:  # noqa: BLE001
        log.warning("[supervisor] LLM erro (%s) — approve (camadas programáticas já validaram)", exc)
        return {"approved": True, "reason": "supervisor llm error (camadas programáticas OK)", "feedback": None, "severity": "ok"}

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
