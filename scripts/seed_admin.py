"""
Cria 1 admin do painel.

Uso:
    python -m scripts.seed_admin admin@local.com senha-forte
    python -m scripts.seed_admin admin@local.com senha-forte "Joao"

Lê SUPABASE_URL + SUPABASE_SERVICE_KEY do env. Faz UPSERT por email.
"""
from __future__ import annotations

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

from panel.auth import hash_password

load_dotenv()


async def main() -> None:
    if len(sys.argv) < 3:
        print("Uso: python -m scripts.seed_admin <email> <senha> [display_name]", file=sys.stderr)
        sys.exit(1)

    email = sys.argv[1].strip().lower()
    password = sys.argv[2]
    display_name = sys.argv[3] if len(sys.argv) > 3 else email.split("@")[0]

    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
    if not url or not key:
        print("ERRO: SUPABASE_URL e SUPABASE_SERVICE_KEY obrigatorios", file=sys.stderr)
        sys.exit(1)

    pwd_hash = hash_password(password)

    async with httpx.AsyncClient(timeout=10.0) as c:
        # Tenta UPDATE primeiro
        r = await c.patch(
            f"{url}/rest/v1/admin_users",
            params={"email": f"eq.{email}"},
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json={
                "password_hash": pwd_hash,
                "display_name": display_name,
            },
        )
        r.raise_for_status()
        data = r.json() or []
        if data:
            print(f"[seed_admin] Atualizou senha de {email}")
            print(f"  id: {data[0]['id']}")
            return

        # INSERT novo
        r2 = await c.post(
            f"{url}/rest/v1/admin_users",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            json={
                "email": email,
                "password_hash": pwd_hash,
                "display_name": display_name,
                "project_ids": [],
            },
        )
        r2.raise_for_status()
        data2 = r2.json() or []
        print(f"[seed_admin] Criou admin {email}")
        if data2:
            print(f"  id: {data2[0]['id']}")


if __name__ == "__main__":
    asyncio.run(main())
