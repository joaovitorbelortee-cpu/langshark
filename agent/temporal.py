"""
Extração de horários explícitos de mensagens em português BR.

Detecta padrões como "5h", "17h", "amanhã", "depois do trabalho", "fim de semana".
Retorna datetime timezone-aware em America/Sao_Paulo.

Determinístico (regex + lookup) — sem LLM. LLM strategist pode complementar
para casos ambíguos via campo `horario_explicito` no prompt.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


BR_TZ = ZoneInfo("America/Sao_Paulo")


def now_br() -> datetime:
    return datetime.now(BR_TZ)


# Período do dia → hora default
_PERIOD_HOURS: dict[str, int] = {
    "manhã": 9, "manha": 9,
    "almoço": 12, "almoco": 12,
    "tarde": 15,
    "noite": 20,
    "madrugada": 2,
    "fim do expediente": 18,
    "depois do trabalho": 19,
    "depois do serviço": 19, "depois do servico": 19,
}

# Dia relativo → offset em dias (ou marker especial)
_DAY_KEYWORDS: dict[str, int | str] = {
    "hoje": 0,
    "amanhã": 1, "amanha": 1,
    "depois de amanhã": 2, "depois de amanha": 2,
    "segunda-feira": "wd-0", "segunda": "wd-0",
    "terça-feira": "wd-1", "terca-feira": "wd-1",
    "terça": "wd-1", "terca": "wd-1",
    "quarta-feira": "wd-2", "quarta": "wd-2",
    "quinta-feira": "wd-3", "quinta": "wd-3",
    "sexta-feira": "wd-4", "sexta": "wd-4",
    "sábado": "wd-5", "sabado": "wd-5",
    "domingo": "wd-6",
    "fim de semana": "weekend",
    "final de semana": "weekend",
}

# Padrões de TEMPO RELATIVO ("daqui X min", "em Y horas") — alta prioridade.
# Tentado ANTES dos absolutos porque "me chama em 17h" é absoluto e "em 5 min"
# é relativo — distintos pelo sufixo.
_RELATIVE_PATTERNS = [
    # "em 3 min", "daqui 5 minutos", "daq 10 mins"
    re.compile(
        r"\b(?:em|daqui(?:\s+a)?|d(?:e)?\s*aqui|daq)\s+(\d{1,4})\s*(?:m|min|mins|minuto|minutos)\b",
        re.IGNORECASE,
    ),
    # "em 1h", "daqui 2 horas"
    re.compile(
        r"\b(?:em|daqui(?:\s+a)?|d(?:e)?\s*aqui|daq)\s+(\d{1,3})\s*(?:h|hora|horas|hs)\b",
        re.IGNORECASE,
    ),
    # "em meia hora", "em uma hora"
    re.compile(
        r"\b(?:em|daqui(?:\s+a)?)\s+(meia|uma|um|dois|duas|tres|tres|cinco|dez)\s+(hora|horas|minuto|minutos)\b",
        re.IGNORECASE,
    ),
]


# Mapeamento texto→número pra padrão extenso ("meia hora", "uma hora")
_WORD_TO_NUMBER: dict[str, float] = {
    "meia": 0.5, "uma": 1, "um": 1,
    "dois": 2, "duas": 2,
    "tres": 3, "três": 3,
    "cinco": 5, "dez": 10,
}


def _resolve_relative_time(text: str) -> int | None:
    """Devolve minutos a partir de agora se mensagem tem tempo RELATIVO.
    None = não tem.

    Exemplos:
      "me chama em 3 min" → 3
      "daqui 1 hora"      → 60
      "em meia hora"      → 30
    """
    # 1) "em N min" / "daqui N min"
    m = _RELATIVE_PATTERNS[0].search(text)
    if m:
        try:
            return max(1, int(m.group(1)))
        except (ValueError, TypeError):
            pass
    # 2) "em N h" / "daqui N horas"
    m = _RELATIVE_PATTERNS[1].search(text)
    if m:
        try:
            return max(1, int(m.group(1)) * 60)
        except (ValueError, TypeError):
            pass
    # 3) "em meia/uma/etc hora"
    m = _RELATIVE_PATTERNS[2].search(text)
    if m:
        word = (m.group(1) or "").lower()
        unit = (m.group(2) or "").lower()
        n = _WORD_TO_NUMBER.get(word)
        if n is not None:
            if unit.startswith("hora"):
                return max(1, int(n * 60))
            if unit.startswith("minuto"):
                return max(1, int(n))
    return None


# Padrões de horário em PT-BR. Ordem importa — tenta cada um até match.
# Cada padrão exige um MARCADOR explícito além do número (h/horas/às/período)
# pra não capturar "5 jogos" ou "tenho 25 anos" como horário.
_TIME_PATTERNS = [
    # 1) "17h30" / "17h" — h como separador/sufixo (mais comum)
    re.compile(
        r"\b(\d{1,2})h(\d{2})?\b(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    # 2) "17:30" / "17:00" — ":" separador clássico
    re.compile(
        r"\b(\d{1,2}):(\d{2})\b(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    # 3) "às 5", "as 17" + opcional h/min/período
    re.compile(
        r"\b(?:às?|as)\s+(\d{1,2})(?:[h:](\d{2}))?(?:\s*h)?\b"
        r"(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    # 4) "17 horas" — sufixo horas/hora explícito
    re.compile(
        r"\b(\d{1,2})(?:[h:](\d{2}))?\s+horas?\b"
        r"(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    # 5) "5 da tarde", "9 da manhã" — período obrigatório
    re.compile(
        r"\b(\d{1,2})(?:[h:](\d{2}))?\s+(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada)\b",
        re.IGNORECASE,
    ),
]


def _resolve_day_offset(text_low: str) -> int | None:
    """Devolve offset de dias da menção temporal mais explícita. None = sem menção."""
    now = now_br()
    for kw, val in _DAY_KEYWORDS.items():
        if kw not in text_low:
            continue
        if isinstance(val, int):
            return val
        if val == "weekend":
            # Próximo sábado (se hoje é sábado/domingo, próximo sábado da semana que vem)
            today_wd = now.weekday()  # 0=seg ... 6=dom
            if today_wd in (5, 6):
                return (5 - today_wd + 7) % 7 or 7
            return (5 - today_wd) % 7
        if val.startswith("wd-"):
            target = int(val.split("-")[1])
            today_wd = now.weekday()
            offset = (target - today_wd) % 7
            return offset if offset > 0 else 7
    return None


def _resolve_hour(text: str) -> tuple[int | None, int]:
    """Extrai hora numérica + minuto. (None, 0) se não encontrou."""
    # Todos os patterns têm formato: (hora, min, período)
    for pat in _TIME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        h_str = m.group(1)
        min_str = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        period = ""
        try:
            period = (m.group(3) or "").lower()
        except IndexError:
            period = ""
        if h_str is None:
            continue
        try:
            h_raw = int(h_str)
        except ValueError:
            continue
        if h_raw > 23:
            continue
        minute = int(min_str or 0)
        if minute > 59:
            continue
        if period in ("tarde", "noite") and h_raw < 12:
            h_raw += 12
        elif period == "madrugada" and h_raw >= 12:
            h_raw -= 12
        return h_raw, minute

    # Não tem hora numérica — tenta período sozinho
    text_low = text.lower()
    for kw, h in _PERIOD_HOURS.items():
        if kw in text_low:
            return h, 0
    return None, 0


def extract_scheduled_time(text: str) -> datetime | None:
    """
    Devolve datetime timezone-aware Brasília se texto mencionar horário absoluto
    OU relativo. None se nenhum horário detectado.

    Exemplos:
      "chego 17h"           → hoje 17:00 (ou amanhã se já passou)
      "amanhã de manhã"     → amanhã 09:00
      "fim de semana"       → próximo sábado 10:00
      "às 8 da noite"       → hoje 20:00 (ou amanhã)
      "depois do trabalho"  → hoje 19:00 (ou amanhã)
      "me chama em 3 min"   → agora + 3 min  ← NEW
      "daqui 1 hora"        → agora + 60 min ← NEW
      "em meia hora"        → agora + 30 min ← NEW
      "olá tudo bem"        → None
    """
    if not text:
        return None

    # 1ª prioridade: tempo RELATIVO ("em 3 min", "daqui 1 hora")
    rel_minutes = _resolve_relative_time(text)
    if rel_minutes is not None:
        return now_br() + timedelta(minutes=rel_minutes)

    text_low = text.lower()
    now = now_br()

    day_offset = _resolve_day_offset(text_low)
    hour, minute = _resolve_hour(text)

    if day_offset is None and hour is None:
        return None  # nada extraível

    if hour is None:
        hour = 10  # default morning se só especificou dia
    if day_offset is None:
        day_offset = 0

    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_offset > 0:
        target += timedelta(days=day_offset)
    elif target <= now + timedelta(minutes=2):
        # Horário já passou (ou está muito próximo) — joga pra amanhã
        target += timedelta(days=1)

    return target


def datetime_to_minutes_from_now(target: datetime) -> int:
    """Converte datetime alvo em delay-minutos. Clamp [1, 10080].
    Min 1 pra suportar "me chama em 3 min" — QStash suporta delay segundos."""
    diff = target - now_br()
    minutes = int(diff.total_seconds() / 60)
    return max(1, min(10080, minutes))
