"""
Cache LRU + TTL pra project_config.

Reduz pressão na Supabase: bot consulta config a cada turno; sem cache, seria
1 query / mensagem. Com TTL 60s + invalidate-on-write, viramos ~1 query / minuto.

Pub/sub Redis cross-worker fica pra F7 (atualmente Railway = 1 worker).
"""
from __future__ import annotations

import time
from typing import Any

from panel.repos import ProjectConfigRepo


class ProjectConfigCache:
    _TTL = 60      # segundos
    _MAX = 256

    def __init__(self) -> None:
        self._store: dict[str, tuple[dict[str, Any], float]] = {}
        self._repo = ProjectConfigRepo()

    async def get(self, project_id: str) -> dict[str, Any]:
        now = time.time()
        cached = self._store.get(project_id)
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
1. WhatsApp = mensagens curtas. MAXIMO 2 bolhas (paragrafos) por resposta.
2. Cada bolha <= 320 caracteres. Quebra com UMA linha em branco.
3. Texto puro. SEM negrito, italico, listas com bullets.
</regras_estritas>

<tags_secretas>
A IA emite tags que o sistema le e remove ANTES de mandar pro cliente:
- [COMPROU] — se o cliente comprou/pagou (comprovante valido). Silencia follow-ups.
- [AGENDAR: N] — minutos ate o proximo follow-up (5-10080). Quente=10-30, Morno=60-180, Frio=360-1440.
- [REACT: emoji] — reage a mensagem do cliente.
- [QUOTE] — usa o "Responder" do WhatsApp citando a ultima mensagem.

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
