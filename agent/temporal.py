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

# Regex hora: "5h", "17h30", "às 5", "9 da manhã", "5 da tarde"
_TIME_RE = re.compile(
    r"""
    \b
    (?:às?|as)?\s*
    (\d{1,2})                       # group 1: hora
    (?::(\d{2}))?                   # group 2: minuto opcional
    \s*(?:h\b|horas?\b|hs?\b|:00\b)?
    (?:\s*(?:da|do|de)\s+
        (manhã|manha|tarde|noite|madrugada))?  # group 3: período
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)


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
    # Tenta regex completo
    m = _TIME_RE.search(text)
    if m:
        h_raw = int(m.group(1))
        if h_raw > 23:
            return None, 0
        minute = int(m.group(2) or 0)
        period = (m.group(3) or "").lower()
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
    Devolve datetime timezone-aware Brasília se texto mencionar horário absoluto.
    None se nenhum horário detectado.

    Exemplos:
      "chego 17h"           → hoje 17:00 (ou amanhã se já passou)
      "amanhã de manhã"     → amanhã 09:00
      "fim de semana"       → próximo sábado 10:00
      "às 8 da noite"       → hoje 20:00 (ou amanhã)
      "depois do trabalho"  → hoje 19:00 (ou amanhã)
      "olá tudo bem"        → None
    """
    if not text:
        return None

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
    """Converte datetime alvo em delay-minutos. Clamp [5, 10080]."""
    diff = target - now_br()
    minutes = int(diff.total_seconds() / 60)
    return max(5, min(10080, minutes))
