"""
Auth do painel: JWT custom + bcrypt + cookie httpOnly.

Sem dependência de Supabase Auth. Único admin seedado via scripts/seed_admin.py.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import bcrypt
from fastapi import Cookie, Depends, HTTPException, Request, Response
from jose import JWTError, jwt

from panel.repos import AdminUsersRepo


_users_repo = AdminUsersRepo()

COOKIE_NAME = "langshark_session"
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24


def _secret() -> str:
    """JWT secret. Reusa WEBHOOK_SECRET (já temos forte 32+ chars)."""
    s = (os.getenv("WEBHOOK_SECRET") or "").strip()
    if not s or len(s) < 16:
        raise RuntimeError("WEBHOOK_SECRET ausente/curta — auth indisponível")
    return s


# ────────────────────────────────────────────────────────────────────
# Hash + verify
# ────────────────────────────────────────────────────────────────────

def _bcrypt_bytes(plain: str) -> bytes:
    """bcrypt aceita no max 72 bytes — corta sem quebrar UTF-8."""
    return plain.encode("utf-8")[:72]


def hash_password(plain: str) -> str:
    hashed = bcrypt.hashpw(_bcrypt_bytes(plain), bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_bcrypt_bytes(plain), hashed.encode("utf-8"))
    except Exception:
        return False


# ────────────────────────────────────────────────────────────────────
# Token JWT
# ────────────────────────────────────────────────────────────────────

def create_token(user: dict[str, Any]) -> str:
    payload = {
        "sub": str(user["id"]),
        "email": user.get("email", ""),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, _secret(), algorithms=[ALGORITHM])
    except JWTError:
        return None


# ────────────────────────────────────────────────────────────────────
# Cookie helpers
# ────────────────────────────────────────────────────────────────────

def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=TOKEN_TTL_HOURS * 3600,
        httponly=True,
        secure=True,        # HTTPS only — Railway sempre HTTPS
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


# ────────────────────────────────────────────────────────────────────
# FastAPI dependencies
# ────────────────────────────────────────────────────────────────────

async def get_current_admin_optional(
    request: Request,
) -> dict[str, Any] | None:
    """Retorna user atual ou None — não levanta. Usar em views Jinja."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    user = await _users_repo.by_id(payload["sub"])
    return user


async def require_admin(
    request: Request,
) -> dict[str, Any]:
    """Dependência pra rotas que exigem login. 401 se anônimo."""
    user = await get_current_admin_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="Auth required")
    return user


# ────────────────────────────────────────────────────────────────────
# Login flow
# ────────────────────────────────────────────────────────────────────

async def authenticate(email: str, password: str) -> dict[str, Any] | None:
    user = await _users_repo.by_email(email.strip().lower())
    if not user:
        return None
    if not verify_password(password, user.get("password_hash", "")):
        return None
    await _users_repo.touch_login(user["id"])
    return user
