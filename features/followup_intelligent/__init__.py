"""Follow-up Intelligent — package portátil.

Public API:
    from followup_intelligent import classify_lead, QStashClient
    from followup_intelligent.temporal import extract_scheduled_time
"""
from .qstash_client import QStashClient
from .strategist import classify_lead, STRATEGIST_MAX_ATTEMPTS
from .temporal import (
    BR_TZ,
    datetime_to_minutes_from_now,
    extract_scheduled_time,
    now_br,
)

__all__ = [
    "classify_lead",
    "QStashClient",
    "STRATEGIST_MAX_ATTEMPTS",
    "extract_scheduled_time",
    "datetime_to_minutes_from_now",
    "now_br",
    "BR_TZ",
]
