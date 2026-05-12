"""
Extração de horários explícitos em mensagens português BR.

Detecta padrões "5h", "17h", "amanhã", "depois do trabalho", "em 3 min", "daqui 1 hora".
Retorna datetime timezone-aware America/Sao_Paulo.

Determinístico (regex + lookup) — sem LLM.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


BR_TZ = ZoneInfo("America/Sao_Paulo")


def now_br() -> datetime:
    return datetime.now(BR_TZ)


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

_RELATIVE_PATTERNS = [
    re.compile(
        r"\b(?:em|daqui(?:\s+a)?|d(?:e)?\s*aqui|daq)\s+(\d{1,4})\s*(?:m|min|mins|minuto|minutos)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:em|daqui(?:\s+a)?|d(?:e)?\s*aqui|daq)\s+(\d{1,3})\s*(?:h|hora|horas|hs)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:em|daqui(?:\s+a)?)\s+(meia|uma|um|dois|duas|tres|três|cinco|dez)\s+(hora|horas|minuto|minutos)\b",
        re.IGNORECASE,
    ),
]

_WORD_TO_NUMBER: dict[str, float] = {
    "meia": 0.5, "uma": 1, "um": 1,
    "dois": 2, "duas": 2,
    "tres": 3, "três": 3,
    "cinco": 5, "dez": 10,
}


def _resolve_relative_time(text: str) -> int | None:
    m = _RELATIVE_PATTERNS[0].search(text)
    if m:
        try:
            return max(1, int(m.group(1)))
        except (ValueError, TypeError):
            pass
    m = _RELATIVE_PATTERNS[1].search(text)
    if m:
        try:
            return max(1, int(m.group(1)) * 60)
        except (ValueError, TypeError):
            pass
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


_TIME_PATTERNS = [
    re.compile(
        r"\b(\d{1,2})h(\d{2})?\b(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2}):(\d{2})\b(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:às?|as)\s+(\d{1,2})(?:[h:](\d{2}))?(?:\s*h)?\b"
        r"(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})(?:[h:](\d{2}))?\s+horas?\b"
        r"(?:\s*(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})(?:[h:](\d{2}))?\s+(?:da|do|de)\s+(manhã|manha|tarde|noite|madrugada)\b",
        re.IGNORECASE,
    ),
]


def _resolve_day_offset(text_low: str) -> int | None:
    now = now_br()
    for kw, val in _DAY_KEYWORDS.items():
        if kw not in text_low:
            continue
        if isinstance(val, int):
            return val
        if val == "weekend":
            today_wd = now.weekday()
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
    for pat in _TIME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        h_str = m.group(1)
        min_str = m.group(2) if m.lastindex and m.lastindex >= 2 else None
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

    text_low = text.lower()
    for kw, h in _PERIOD_HOURS.items():
        if kw in text_low:
            return h, 0
    return None, 0


def extract_scheduled_time(text: str) -> datetime | None:
    """
    Retorna datetime timezone-aware Brasília se texto menciona horário.
    None se nenhum detectado.

    Exemplos:
      "chego 17h"           → hoje 17:00 (ou amanhã se já passou)
      "amanhã de manhã"     → amanhã 09:00
      "fim de semana"       → próximo sábado 10:00
      "em 3 min"            → agora + 3 min
      "daqui 1 hora"        → agora + 60 min
    """
    if not text:
        return None

    rel_minutes = _resolve_relative_time(text)
    if rel_minutes is not None:
        return now_br() + timedelta(minutes=rel_minutes)

    text_low = text.lower()
    now = now_br()

    day_offset = _resolve_day_offset(text_low)
    hour, minute = _resolve_hour(text)

    if day_offset is None and hour is None:
        return None

    if hour is None:
        hour = 10
    if day_offset is None:
        day_offset = 0

    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_offset > 0:
        target += timedelta(days=day_offset)
    elif target <= now + timedelta(minutes=2):
        target += timedelta(days=1)

    return target


def datetime_to_minutes_from_now(target: datetime) -> int:
    """Converte datetime alvo em delay-minutos. Clamp [1, 10080]."""
    diff = target - now_br()
    minutes = int(diff.total_seconds() / 60)
    return max(1, min(10080, minutes))
