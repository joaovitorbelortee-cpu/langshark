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
    """Parágrafos com conteúdo substancial mantêm separação visual."""
    text = "Esse é o primeiro parágrafo com algum conteúdo de verdade.\n\nE esse aqui é o segundo bloco totalmente separado."
    chunks = chunk_for_whatsapp(text, max_bubbles=3, max_chars=140)
    assert len(chunks) == 2


def test_chunk_caps_at_max_bubbles():
    """Excesso de parágrafos é concatenado na última bolha pra não passar do cap."""
    text = "Bloco um com tamanho normal.\n\nBloco dois com tamanho normal.\n\nBloco três com tamanho normal.\n\nBloco quatro com tamanho normal."
    chunks = chunk_for_whatsapp(text, max_bubbles=2, max_chars=320)
    assert len(chunks) == 2
    # último concatena os 3 últimos
    assert "Bloco dois" in chunks[1] and "Bloco três" in chunks[1] and "Bloco quatro" in chunks[1]


def test_chunk_long_paragraph_splits_sentences():
    long = "Frase um. " * 60   # ~600 chars
    chunks = chunk_for_whatsapp(long, max_bubbles=3, max_chars=200)
    assert len(chunks) >= 2
    assert all(len(c) <= 250 for c in chunks)  # margem


def test_chunk_preserves_url():
    """URLs nunca podem ser quebradas no meio."""
    text = "Aqui está o link de pagamento: https://pay.example.com/abc/123/very-long-checkout-path-id"
    chunks = chunk_for_whatsapp(text, max_bubbles=3, max_chars=60)
    # Mesmo com max_chars baixo, URL deve aparecer inteira em ALGUMA bolha
    joined = " ".join(chunks)
    assert "https://pay.example.com/abc/123/very-long-checkout-path-id" in joined
    # Nenhuma bolha pode ter URL pela metade
    for c in chunks:
        if "https://" in c:
            assert "pay.example.com" in c
            assert "checkout-path-id" in c


def test_chunk_preserves_pix_uuid():
    """Chaves PIX UUID nunca quebram."""
    text = "Te mando o pix agora. Chave: 12345678-abcd-1234-efgh-123456789012 confirma quando cair?"
    chunks = chunk_for_whatsapp(text, max_bubbles=3, max_chars=50)
    joined = " ".join(chunks)
    assert "12345678-abcd-1234-efgh-123456789012" in joined


def test_chunk_preserves_money_ranges():
    """Valores monetários e ranges (R$X a R$Y) ficam inteiros."""
    text = "Tem 3 planos: Básico R$40 a R$60, Premium R$80 a R$120, Ultimate R$150 por mês."
    chunks = chunk_for_whatsapp(text, max_bubbles=3, max_chars=60)
    joined = " ".join(chunks)
    # Valores agrupados nunca podem ser cortados
    assert "R$40 a R$60" in joined
    assert "R$80 a R$120" in joined


def test_chunk_humanizes_long_single_sentence():
    """Usuário pediu: split em mais bolhas pra parecer humano."""
    text = "Entendo, faz sentido você achar caro. Mas pensa que um jogo novo custa de R$250 a R$300. Com o Game Pass, você tem acesso a mais de 400 jogos por muito menos."
    chunks = chunk_for_whatsapp(text)  # defaults: 3 bolhas, 140 chars
    assert 2 <= len(chunks) <= 3
    # Cada bolha respeita max_chars
    for c in chunks:
        assert len(c) <= 180  # margem
    # R$ ranges preservados
    joined = " ".join(chunks)
    assert "R$250 a R$300" in joined


def test_chunk_merges_tiny_with_neighbor():
    """Bolhas minúsculas (<25 chars) mergeiam com vizinha pra evitar 'ok.' sozinho."""
    text = "Beleza. Vou te passar o link agora mesmo se você confirmar."
    chunks = chunk_for_whatsapp(text)
    # "Beleza." (7 chars) tem que ter merge com algo
    assert all(len(c) >= 20 for c in chunks)


def test_parse_flow_tag_present():
    name, cleaned = parse_flow_tag("Manda esse fluxo: [FLOW: boas_vindas] hoje")
    assert name == "boas_vindas"
    assert "[FLOW" not in cleaned


def test_parse_flow_tag_absent():
    name, cleaned = parse_flow_tag("oi tudo bem")
    assert name is None
    assert cleaned == "oi tudo bem"
