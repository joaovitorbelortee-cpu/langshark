"""
Cache LRU + TTL pra project_config.

Reduz pressão na Supabase: bot consulta config a cada turno; sem cache, seria
1 query / mensagem. Com TTL 60s + invalidate-on-write, viramos ~1 query / minuto.

Pub/sub Redis cross-worker fica pra F7 (atualmente Railway = 1 worker).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from panel.repos import ProjectConfigRepo


class ProjectConfigCache:
    _TTL = 60      # segundos
    _MAX = 256

    def __init__(self) -> None:
        self._store: dict[str, tuple[dict[str, Any], float]] = {}
        self._repo = ProjectConfigRepo()
        self._lock = asyncio.Lock()

    async def get(self, project_id: str) -> dict[str, Any]:
        now = time.monotonic()
        cached = self._store.get(project_id)
        if cached and cached[1] > now:
            return cached[0]

        # Single-flight: evita cache stampede quando vários webhooks chegam ao mesmo tempo.
        async with self._lock:
            cached = self._store.get(project_id)
            now = time.monotonic()
            if cached and cached[1] > now:
                return cached[0]
            cfg = await self._repo.fetch(project_id) or {}
            self._store[project_id] = (cfg, now + self._TTL)
            if len(self._store) > self._MAX:
                self._evict_oldest()
            return cfg

    def invalidate(self, project_id: str) -> None:
        self._store.pop(project_id, None)

    def _evict_oldest(self) -> None:
        # Remove entrada com expires_at mais próximo do passado
        if not self._store:
            return
        oldest = min(self._store.items(), key=lambda kv: kv[1][1])
        self._store.pop(oldest[0], None)


_singleton: ProjectConfigCache | None = None


def get_project_config_cache() -> ProjectConfigCache:
    global _singleton
    if _singleton is None:
        _singleton = ProjectConfigCache()
    return _singleton


# ────────────────────────────────────────────────────────────────────
# compose_system_prompt — fonte única de verdade do prompt do bot
# ────────────────────────────────────────────────────────────────────

ORDER = [
    "company_info",
    "prices",
    "parameters",
    "priority_situations",
    "knowledge_base",
]

# Footer técnico — sempre injetado, não editável pelo painel.
TAGS_FOOTER = """<regras_estritas>
1. WhatsApp = mensagens curtas e humanas. MAXIMO 3 bolhas por resposta.
2. Cada bolha 60-140 caracteres (1-2 frases). NUNCA mande paredão de texto.
3. Quebra entre bolhas com UMA linha em branco. Sentencas separadas, ritmo natural.
4. Texto puro. SEM negrito, italico, listas com bullets.
5. NUNCA quebre links, chaves PIX, R$ valores, emails ou telefones — sistema protege automatico.
</regras_estritas>

<tags_secretas>
A IA emite tags que o sistema le e remove ANTES de mandar pro cliente:

- [COMPROU] — cliente pagou (comprovante valido). Silencia follow-ups.

- [AGENDAR: N] — minutos ate o proximo follow-up (5-10080).
  Quente=10-30, Morno=60-180, Frio=360-1440.

- [REACT: emoji] — bot reage a mensagem do cliente com 1 emoji (no notification).
  USE COM PARCIMONIA — humano reage uns 20-30% das mensagens, nao toda hora.
  USE QUANDO:
    * Cliente mandou algo ENGRAÇADO          → 😂 🤣
    * Cliente mandou algo TRISTE/frustrado   → 😢 🥺
    * Cliente mandou algo EMPOLGANTE         → 🔥 💯 🚀
    * Cliente mandou algo SURPRESA           → 😮 🤯
    * Cliente AGRADECEU ou ELOGIOU           → 🙏 ❤️ 👏
    * Cliente CONFIRMOU compra/decisao       → 🎉 ✅ 👍
  NUNCA REAJA 2 mensagens seguidas — sistema bloqueia tambem.
  Em duvida, NAO REAJA.

- [QUOTE] — bot responde citando a mensagem do cliente (botao "Responder" WhatsApp).
  REGRA AUTOMATICA: sistema sempre cita quando a msg do cliente contem "?"
  (pergunta direta). Voce nao precisa emitir [QUOTE] nesses casos — eh automatico.
  USE [QUOTE] manualmente nestes outros casos (~15-25% das respostas):
    * Conversa pulou de tema e voce quer ANCORAR no que ele disse.
    * Cliente mandou multiplas perguntas/afirmacoes — quote a que voce responde.
    * Resposta poderia confundir sem o contexto da msg dele.
  NAO USE em saudacao, "ok", "valeu" — nao tem contexto pra ancorar.
  NAO USE 2 turnos seguidos (sistema bloqueia com cooldown automatico).

Sempre encerre com [AGENDAR: N], a menos que [COMPROU] esteja presente.
</tags_secretas>"""


def compose_system_prompt(cfg: dict[str, Any]) -> str:
    """Concatena seções editáveis + footer técnico fixo. Retorna string vazia se cfg vazio."""
    sections = (cfg or {}).get("brain_sections") or {}
    parts: list[str] = []
    for key in ORDER:
        s = sections.get(key) or {}
        content = (s.get("content") or "").strip()
        if content:
            title = s.get("title", key)
            parts.append(f"# {title}\n{content}")
    if not parts:
        return ""  # caller decide fallback (SALES_SYSTEM hardcoded)
    parts.append(TAGS_FOOTER)
    return "\n\n".join(parts)
