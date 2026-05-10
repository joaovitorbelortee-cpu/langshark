"""Testes unitários puros — parsers e chunking. Sem rede, sem LLM."""
from __future__ import annotations

from agent.flows import parse_flow_tag
from agent.tools import chunk_for_whatsapp, parse_tags


def test_parse_tags_extracts_compraou():
    parsed = parse_tags("Show, recebi seu pix! [COMPROU]")
    assert parsed.has_converted is True
    assert "[COMPROU]" not in parsed.text
    assert parsed.text.strip() == "Show, recebi seu pix!"


def test_parse_tags_agendar_clamped():
    parsed = parse_tags("até depois [AGENDAR: 999999]")
    assert parsed.schedule_minutes == 10080  # max


def test_parse_tags_agendar_min():
    parsed = parse_tags("[AGENDAR: 1]")
    assert parsed.schedule_minutes == 5  # min


def test_parse_tags_react_emoji():
    parsed = parse_tags("Obrigado! [REACT:🔥]")
    assert parsed.react_emoji == "🔥"
    assert "[REACT" not in parsed.text


def test_parse_tags_quote():
    parsed = parse_tags("respondendo [QUOTE]")
    assert parsed.quote_previous is True


def test_parse_tags_all_combined():
    raw = "Beleza! [REACT:👍] [QUOTE] [AGENDAR: 60]"
    p = parse_tags(raw)
    assert p.react_emoji == "👍"
    assert p.quote_previous is True
    assert p.schedule_minutes == 60
    assert p.has_converted is False
    assert p.text.strip() == "Beleza!"


def test_chunk_short_text_one_bubble():
    chunks = chunk_for_whatsapp("Oi tudo bem?")
    assert len(chunks) == 1
    assert chunks[0] == "Oi tudo bem?"


def test_chunk_paragraph_split():
    text = "Primeira parte.\n\nSegunda parte."
    chunks = chunk_for_whatsapp(text, max_bubbles=2, max_chars=320)
    assert len(chunks) == 2


def test_chunk_caps_at_max_bubbles():
    text = "p1\n\np2\n\np3\n\np4"
    chunks = chunk_for_whatsapp(text, max_bubbles=2, max_chars=320)
    assert len(chunks) == 2
    # último concatena os 3 últimos
    assert "p2" in chunks[1] and "p3" in chunks[1] and "p4" in chunks[1]


def test_chunk_long_paragraph_splits_sentences():
    long = "Frase um. " * 60   # ~600 chars
    chunks = chunk_for_whatsapp(long, max_bubbles=3, max_chars=200)
    assert len(chunks) >= 2
    assert all(len(c) <= 250 for c in chunks)  # margem


def test_parse_flow_tag_present():
    name, cleaned = parse_flow_tag("Manda esse fluxo: [FLOW: boas_vindas] hoje")
    assert name == "boas_vindas"
    assert "[FLOW" not in cleaned


def test_parse_flow_tag_absent():
    name, cleaned = parse_flow_tag("oi tudo bem")
    assert name is None
    assert cleaned == "oi tudo bem"
