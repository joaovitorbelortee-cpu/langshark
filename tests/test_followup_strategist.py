"""
Tests pro Follow-up Strategist + Temporal extraction.

Focus em unit-test puro: extração regex (sem LLM), validação JSON,
clamps por temperatura, escalação por tentativas.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest


# ────────────────────────────────────────────────────────────────────
# Temporal extraction — regex puro, determinístico
# ────────────────────────────────────────────────────────────────────

def test_temporal_extracts_hour_only():
    """'5h', 'às 17h' → datetime hoje na hora dada."""
    from agent.temporal import extract_scheduled_time, now_br
    now = now_br()
    # Use uma hora claramente futura
    future_h = (now.hour + 3) % 24
    dt = extract_scheduled_time(f"chego {future_h}h")
    assert dt is not None
    assert dt.hour == future_h


def test_temporal_extracts_period():
    """'tarde' → 15h, 'noite' → 20h, 'manhã' → 9h."""
    from agent.temporal import extract_scheduled_time
    assert extract_scheduled_time("te falo de tarde").hour == 15
    assert extract_scheduled_time("a noite").hour == 20
    assert extract_scheduled_time("amanha de manha").hour == 9


def test_temporal_extracts_tomorrow():
    """'amanhã' → +1 dia."""
    from agent.temporal import extract_scheduled_time, now_br
    now = now_br()
    dt = extract_scheduled_time("amanha de manha")
    assert dt is not None
    assert dt.date() == (now + timedelta(days=1)).date()
    assert dt.hour == 9


def test_temporal_extracts_weekend():
    """'fim de semana' → próximo sábado."""
    from agent.temporal import extract_scheduled_time
    dt = extract_scheduled_time("a gente fala fim de semana")
    assert dt is not None
    assert dt.weekday() == 5  # sábado


def test_temporal_extracts_specific_weekday():
    """'sexta', 'segunda' → próxima daquele weekday."""
    from agent.temporal import extract_scheduled_time
    dt = extract_scheduled_time("falo com vc segunda")
    assert dt is not None
    assert dt.weekday() == 0  # segunda


def test_temporal_combines_day_and_hour():
    """'amanhã às 14h' → amanhã 14:00."""
    from agent.temporal import extract_scheduled_time, now_br
    now = now_br()
    dt = extract_scheduled_time("amanha as 14h")
    assert dt is not None
    assert dt.date() == (now + timedelta(days=1)).date()
    assert dt.hour == 14


def test_temporal_no_match_returns_none():
    """Texto sem horário → None."""
    from agent.temporal import extract_scheduled_time
    assert extract_scheduled_time("ola tudo bem?") is None
    assert extract_scheduled_time("vou pensar e te falo") is None
    assert extract_scheduled_time("") is None


def test_temporal_past_time_rolls_to_tomorrow():
    """Hora já passou hoje → joga pra amanhã automaticamente."""
    from agent.temporal import extract_scheduled_time, now_br
    now = now_br()
    if now.hour < 3:
        pytest.skip("Madrugada — sem hora passada pra testar")
    # Hora 1 da manhã sempre é passada (excepto entre 0-1h)
    dt = extract_scheduled_time("te falo as 1h")
    assert dt is not None
    # Deve ser amanhã (ou hoje se ainda for madrugada)
    assert dt > now


def test_minutes_calc():
    """datetime_to_minutes_from_now retorna delta em minutos."""
    from agent.temporal import datetime_to_minutes_from_now, now_br
    target = now_br() + timedelta(minutes=90)
    mins = datetime_to_minutes_from_now(target)
    assert 85 <= mins <= 95  # margem


def test_minutes_clamps_low():
    """Hora muito próxima é clamped pro mínimo (5 min)."""
    from agent.temporal import datetime_to_minutes_from_now, now_br
    target = now_br() + timedelta(seconds=10)
    assert datetime_to_minutes_from_now(target) == 5


def test_minutes_clamps_high():
    """Hora muito distante (> 1 semana) é clamped pro máximo."""
    from agent.temporal import datetime_to_minutes_from_now, now_br
    target = now_br() + timedelta(days=30)
    assert datetime_to_minutes_from_now(target) == 10080


# ────────────────────────────────────────────────────────────────────
# Strategist validation — sanitiza output do LLM
# ────────────────────────────────────────────────────────────────────

def test_validate_hot_clamps_to_60min():
    """HOT decision com agendar_minutos=500 é clamped pra 60."""
    from agent.follow_up_strategist import _validate_decision
    out = _validate_decision(
        {"temperatura": "HOT", "agendar_minutos": 500, "abordagem": "commitment"},
        regex_dt=None,
        attempts_made=0,
    )
    assert out["temperatura"] == "HOT"
    assert out["agendar_minutos"] <= 60


def test_validate_warm_clamps_to_range():
    """WARM agendar_minutos clamped 60-180."""
    from agent.follow_up_strategist import _validate_decision
    out = _validate_decision(
        {"temperatura": "WARM", "agendar_minutos": 10, "abordagem": "valor"},
        regex_dt=None, attempts_made=0,
    )
    assert 60 <= out["agendar_minutos"] <= 180


def test_validate_cold_clamps_to_12h_24h():
    """COLD agendar_minutos clamped 720-1440 (12-24h)."""
    from agent.follow_up_strategist import _validate_decision
    out = _validate_decision(
        {"temperatura": "COLD", "agendar_minutos": 30, "abordagem": "valor"},
        regex_dt=None, attempts_made=0,
    )
    assert 720 <= out["agendar_minutos"] <= 1440


def test_validate_stop_sets_killswitch():
    """STOP sempre seta killswitch=True e minutos=0."""
    from agent.follow_up_strategist import _validate_decision
    out = _validate_decision(
        {"temperatura": "STOP", "agendar_minutos": 60, "abordagem": "valor"},
        regex_dt=None, attempts_made=0,
    )
    assert out["temperatura"] == "STOP"
    assert out["killswitch_permanent"] is True
    assert out["agendar_minutos"] == 0


def test_validate_invalid_temp_defaults_warm():
    """Temperatura inválida do LLM → WARM safe default."""
    from agent.follow_up_strategist import _validate_decision
    out = _validate_decision(
        {"temperatura": "MORNO", "agendar_minutos": 100, "abordagem": "x"},
        regex_dt=None, attempts_made=0,
    )
    assert out["temperatura"] == "WARM"
    assert out["abordagem"] == "valor"  # default abordagem também


def test_validate_attempts_3_escalates_warm():
    """3+ tentativas em WARM → multiplica delay ×2, força VALOR."""
    from agent.follow_up_strategist import _validate_decision
    out = _validate_decision(
        {"temperatura": "WARM", "agendar_minutos": 90, "abordagem": "commitment"},
        regex_dt=None, attempts_made=3,
    )
    # 90 * 2 = 180, clamped to 60-180 max → 180
    assert out["agendar_minutos"] == 180
    assert out["abordagem"] == "valor"


def test_validate_attempts_max_via_classify_returns_stop():
    """Atingir MAX_ATTEMPTS no classify_lead retorna STOP+killswitch sem chamar LLM."""
    import asyncio

    from agent.follow_up_strategist import classify_lead, STRATEGIST_MAX_ATTEMPTS

    async def go():
        out = await classify_lead(
            messages=[], last_user_message="oi",
            attempts_made=STRATEGIST_MAX_ATTEMPTS,
        )
        assert out["temperatura"] == "STOP"
        assert out["killswitch_permanent"] is True
        assert out["agendar_minutos"] == 0

    asyncio.run(go())


def test_validate_scheduled_uses_horario():
    """SCHEDULED com horário ISO → calcula minutos do horário."""
    from datetime import datetime, timedelta

    from agent.follow_up_strategist import _validate_decision
    from agent.temporal import BR_TZ, now_br
    future = (now_br() + timedelta(hours=3)).replace(microsecond=0)
    out = _validate_decision(
        {
            "temperatura": "SCHEDULED",
            "horario_explicito": future.isoformat(),
            "agendar_minutos": 0,  # LLM passou 0, código calcula
            "abordagem": "commitment",
        },
        regex_dt=None, attempts_made=0,
    )
    assert out["temperatura"] == "SCHEDULED"
    # 3h = 180min — margem de erro
    assert 170 <= out["agendar_minutos"] <= 195


def test_fallback_decision_with_regex_dt():
    """Fallback (LLM offline) usa regex_dt como SCHEDULED."""
    from datetime import timedelta

    from agent.follow_up_strategist import _fallback_decision
    from agent.temporal import now_br
    future = now_br() + timedelta(hours=2)
    out = _fallback_decision(attempts_made=1, regex_dt=future)
    assert out["temperatura"] == "SCHEDULED"
    assert out["horario_explicito"] is not None


def test_fallback_decision_warm_default():
    """Fallback sem regex_dt → WARM 90min."""
    from agent.follow_up_strategist import _fallback_decision
    out = _fallback_decision(attempts_made=0, regex_dt=None)
    assert out["temperatura"] == "WARM"
    assert out["agendar_minutos"] == 90
    assert out["killswitch_permanent"] is False


def test_fallback_attempts_3_increases_delay():
    """Fallback com 3+ tentativas → delay 240min (4h)."""
    from agent.follow_up_strategist import _fallback_decision
    out = _fallback_decision(attempts_made=3, regex_dt=None)
    assert out["agendar_minutos"] == 240


# ────────────────────────────────────────────────────────────────────
# Redis attempt counter
# ────────────────────────────────────────────────────────────────────

async def test_attempt_counter_increments():
    """INCR + get retorna sequência crescente."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    assert await store.get_followup_attempts("inst", "5511") == 0
    n1 = await store.increment_followup_attempts("inst", "5511")
    n2 = await store.increment_followup_attempts("inst", "5511")
    n3 = await store.increment_followup_attempts("inst", "5511")
    assert n1 == 1
    assert n2 == 2
    assert n3 == 3
    assert await store.get_followup_attempts("inst", "5511") == 3


async def test_attempt_counter_reset():
    """Lead respondeu → DEL zera contador."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    for _ in range(5):
        await store.increment_followup_attempts("inst", "5511")
    assert await store.get_followup_attempts("inst", "5511") == 5
    await store.reset_followup_attempts("inst", "5511")
    assert await store.get_followup_attempts("inst", "5511") == 0


async def test_attempt_counter_per_phone_isolated():
    """Counters de phones diferentes são independentes."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    await store.increment_followup_attempts("inst", "111")
    await store.increment_followup_attempts("inst", "111")
    await store.increment_followup_attempts("inst", "222")
    assert await store.get_followup_attempts("inst", "111") == 2
    assert await store.get_followup_attempts("inst", "222") == 1
    assert await store.get_followup_attempts("inst", "333") == 0


# ────────────────────────────────────────────────────────────────────
# Lead status registry — alimenta painel Reconquista
# ────────────────────────────────────────────────────────────────────

async def test_lead_status_roundtrip():
    """set_lead_status + get_lead_status preservam JSON aninhado."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    snapshot = {
        "project_id": "padrao",
        "instance": "botzap",
        "phone": "5511999999999",
        "temperatura": "HOT",
        "abordagem": "commitment",
        "razao": "escolheu plano anual",
        "agendar_minutos": 30,
        "attempts_made": 0,
        "last_decision_at": "2026-05-11T12:00:00+00:00",
        "next_followup_at": "2026-05-11T12:30:00+00:00",
        "killswitch_permanent": False,
    }
    await store.set_lead_status("botzap", "5511999999999", snapshot)
    got = await store.get_lead_status("botzap", "5511999999999")
    assert got is not None
    assert got["temperatura"] == "HOT"
    assert got["agendar_minutos"] == 30


async def test_lead_status_missing_returns_none():
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    assert await store.get_lead_status("inst", "1234") is None


async def test_lead_status_list_orders_by_recency():
    """list_lead_statuses retorna mais recente primeiro."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    await store.set_lead_status("inst", "111", {
        "phone": "111", "temperatura": "WARM",
        "last_decision_at": "2026-05-11T10:00:00+00:00",
    })
    await store.set_lead_status("inst", "222", {
        "phone": "222", "temperatura": "HOT",
        "last_decision_at": "2026-05-11T15:00:00+00:00",
    })
    await store.set_lead_status("inst", "333", {
        "phone": "333", "temperatura": "COLD",
        "last_decision_at": "2026-05-11T12:00:00+00:00",
    })
    leads = await store.list_lead_statuses()
    assert len(leads) == 3
    # Mais recente primeiro: 222 (15h), 333 (12h), 111 (10h)
    assert leads[0]["phone"] == "222"
    assert leads[1]["phone"] == "333"
    assert leads[2]["phone"] == "111"


async def test_lead_status_list_dedups_repeated_updates():
    """Múltiplos sets na mesma chave → apenas 1 entry na lista."""
    from memory.redis_store import RedisStore
    store = RedisStore(url="", token="")
    for i in range(5):
        await store.set_lead_status("inst", "111", {
            "phone": "111", "temperatura": "WARM",
            "last_decision_at": f"2026-05-11T1{i}:00:00+00:00",
        })
    leads = await store.list_lead_statuses()
    # Apesar de 5 sets, apenas 1 entry (dedup por chave)
    assert len(leads) == 1
    assert leads[0]["phone"] == "111"
