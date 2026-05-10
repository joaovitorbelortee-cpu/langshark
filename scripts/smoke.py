"""
Smoke test E2E contra um deploy rodando.

Uso:
    python -m scripts.smoke                              # localhost
    python -m scripts.smoke --url https://x.railway.app  # prod
    python -m scripts.smoke --secret meu-secret          # auth do webhook

Checklist:
  1. GET /health  -> 200 com flags esperadas
  2. POST /webhook (auth inválida) -> 401
  3. POST /webhook (event ignorado) -> 200 ok+skipped
  4. POST /webhook (mensagem real fake) -> 200 processed
  5. POST /api/trigger-followup (sem instance) -> 400
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx


async def step(label: str, ok: bool, detail: str = "") -> bool:
    sym = "[ok]" if ok else "[x]"
    print(f"  {sym} {label}" + (f"  ({detail})" if detail else ""))
    return ok


async def run(base_url: str, secret: str) -> int:
    print(f"\n-> Smoke test against {base_url}\n")
    failures = 0
    timeout = httpx.Timeout(15.0, connect=5.0)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        # 1. /health
        try:
            r = await client.get("/health")
            ok = r.status_code == 200 and r.json().get("ok") is True
            await step("/health 200", ok, f"body={r.json()}")
            if not ok:
                failures += 1
        except Exception as exc:
            await step("/health 200", False, f"exception {exc}")
            failures += 1

        # 2. /webhook unauthorized
        try:
            r = await client.post(
                "/webhook",
                headers={"apikey": "wrong-secret"},
                json={"event": "messages.upsert", "data": {}},
            )
            ok = r.status_code == 401
            await step("/webhook 401 sem auth", ok, f"got {r.status_code}")
            if not ok:
                failures += 1
        except Exception as exc:
            await step("/webhook 401 sem auth", False, str(exc))
            failures += 1

        # 3. /webhook event ignorado
        try:
            r = await client.post(
                "/webhook",
                headers={"apikey": secret},
                json={"event": "connection.update", "data": {"state": "open"}, "instance": "smoke"},
            )
            ok = r.status_code == 200 and r.json().get("ok") is True
            await step("/webhook 200 connection.update", ok, f"body={r.json()}")
            if not ok:
                failures += 1
        except Exception as exc:
            await step("/webhook 200 connection.update", False, str(exc))
            failures += 1

        # 4. /webhook fromMe = skip
        try:
            r = await client.post(
                "/webhook",
                headers={"apikey": secret},
                json={
                    "event": "messages.upsert",
                    "instance": "smoke",
                    "data": {
                        "key": {"fromMe": True, "remoteJid": "5511999999999@s.whatsapp.net", "id": "X"},
                        "message": {"conversation": "test"},
                    },
                },
            )
            ok = r.status_code == 200 and "fromMe" in json.dumps(r.json())
            await step("/webhook 200 fromMe skip", ok, f"body={r.json()}")
            if not ok:
                failures += 1
        except Exception as exc:
            await step("/webhook 200 fromMe skip", False, str(exc))
            failures += 1

        # 5. /api/trigger-followup sem corpo válido
        try:
            r = await client.post(
                "/api/trigger-followup",
                headers={"apikey": secret},
                json={"phone": "5511999999999"},  # falta instance_name
            )
            ok = r.status_code == 400
            await step("/api/trigger-followup 400 sem instance", ok, f"got {r.status_code}")
            if not ok:
                failures += 1
        except Exception as exc:
            await step("/api/trigger-followup 400 sem instance", False, str(exc))
            failures += 1

    print(f"\n{'[FAIL]' if failures else '[PASS]'}: {failures} failure(s)\n")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("SMOKE_URL", "http://localhost:8000"))
    parser.add_argument("--secret", default=os.getenv("WEBHOOK_SECRET", "troque-isso-por-um-segredo-forte"))
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.url, args.secret)))


if __name__ == "__main__":
    main()
