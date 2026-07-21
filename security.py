"""JWT token generation / verification and password hashing utilities."""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
from dotenv import load_dotenv
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from models import User
from services.log_context import bind_log_context

load_dotenv()

_PLACEHOLDER_SECRETS = {
    "",
    "change-me",
    "change-me-to-a-random-string",
    "secret",
    "jwt-secret",
    "your-secret-key",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def validate_jwt_secret(secret: Optional[str]) -> str:
    """Reject missing / placeholder / short JWT secrets before the app serves traffic."""
    value = (secret or "").strip()
    if value.lower() in _PLACEHOLDER_SECRETS or len(value) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY is missing or too weak. "
            "Set a random secret with at least 32 characters "
            "(for example: openssl rand -hex 32)."
        )
    return value


SECRET_KEY = validate_jwt_secret(os.getenv("JWT_SECRET_KEY"))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
COOKIE_SECURE = _env_flag("COOKIE_SECURE", default=False)
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax").strip().lower() or "lax"
if COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    COOKIE_SAMESITE = "lax"
ACCESS_COOKIE_NAME = "access_token"
REFRESH_COOKIE_NAME = "refresh_token"
CSRF_COOKIE_NAME = "csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"

bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _encode_token(payload: dict[str, Any]) -> str:
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return _encode_token(payload)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_hex(16),
    }
    return _encode_token(payload)


def decode_token(token: str, *, expected_type: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录状态无效或已过期",
        ) from exc

    if payload.get("type") != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录状态无效或已过期",
        )
    user_id = payload.get("sub")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录状态无效或已过期",
        )
    return payload


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_auth_cookies(
    response: Response,
    *,
    access_token: str,
    refresh_token: str,
    csrf_token: str,
) -> None:
    common = {
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE,
        "path": "/",
    }
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **common,
    )
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        **common,
    )
    # Readable by JS so the frontend can mirror it into X-CSRF-Token.
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )


def clear_auth_cookies(response: Response) -> None:
    for name in (ACCESS_COOKIE_NAME, REFRESH_COOKIE_NAME, CSRF_COOKIE_NAME):
        response.delete_cookie(key=name, path="/")


def _extract_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Optional[str]:
    if credentials is None:
        return None
    if credentials.scheme.lower() != "bearer":
        return None
    token = (credentials.credentials or "").strip()
    return token or None


def _request_uses_cookie_auth(request: Request, bearer_token: Optional[str]) -> bool:
    if bearer_token:
        return False
    return bool(request.cookies.get(ACCESS_COOKIE_NAME) or request.cookies.get(REFRESH_COOKIE_NAME))


def validate_csrf(request: Request) -> None:
    """Double-submit cookie CSRF check for cookie-authenticated mutating requests."""
    if request.method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    header_token = request.headers.get(CSRF_HEADER_NAME, "")
    if not cookie_token or not header_token or not secrets.compare_digest(cookie_token, header_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF 校验失败",
        )


def _load_user(user_id: str) -> User:
    with Session(engine) as db:
        user = db.scalar(select(User).where(User.id == user_id))
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登录状态无效或已过期",
            )
        # Detach-safe copy of attributes used by callers.
        db.expunge(user)
        return user


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> User:
    bearer_token = _extract_bearer_token(credentials)
    cookie_token = request.cookies.get(ACCESS_COOKIE_NAME)
    token = bearer_token or cookie_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或登录已过期",
        )

    if _request_uses_cookie_auth(request, bearer_token):
        validate_csrf(request)

    payload = decode_token(token, expected_type="access")
    user = _load_user(payload["sub"])
    request.state.user_id = user.id
    bind_log_context(user_id=user.id)
    return user


def issue_token_pair(user_id: str) -> dict[str, Any]:
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)
    csrf_token = new_csrf_token()
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "csrf_token": csrf_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }
