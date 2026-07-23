"""Authentication routes."""

from __future__ import annotations

import secrets
import string
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import engine
from models import InviteCode, User
from security import (
    clear_auth_cookies,
    decode_token,
    get_current_user,
    hash_password,
    issue_token_pair,
    set_auth_cookies,
    validate_csrf,
    verify_password,
    REFRESH_COOKIE_NAME,
)
from settings import BOOTSTRAP_ADMIN_PHONE, INVITE_CODE_TTL_DAYS

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_FAILURE_DETAIL = "手机号或密码错误"
_INVITE_ALPHABET = string.ascii_uppercase + string.digits


class PhoneCheckRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)


class LoginRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=32)


class RegisterRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=32)
    invite_code: str = Field(..., min_length=6, max_length=16)


class RefreshRequest(BaseModel):
    refresh_token: Optional[str] = None


def build_registration_user(phone: str, password: str) -> User:
    """Create a user with an identifier that cannot collide on phone suffixes."""
    user_id = str(uuid.uuid4())
    return User(
        id=user_id,
        username=f"user_{user_id.replace('-', '')}",
        phone=phone,
        hashed_password=hash_password(password),
    )


def _auth_response_payload(user: User, tokens: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user.id,
        "phone": user.phone,
        "token_type": tokens["token_type"],
        "expires_in": tokens["expires_in"],
        # Kept for backward-compatible clients that still send Authorization.
        # Browser clients should rely on HttpOnly cookies.
        "token": tokens["access_token"],
        "access_token": tokens["access_token"],
    }


def _set_session_cookies(response: Response, tokens: dict[str, Any]) -> None:
    set_auth_cookies(
        response,
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        csrf_token=tokens["csrf_token"],
    )


def _load_user(user_id: str) -> User:
    with Session(engine) as db:
        user = db.scalar(select(User).where(User.id == user_id))
        if user is None:
            raise HTTPException(status_code=401, detail="登录状态无效或已过期")
        db.expunge(user)
        return user


@router.post("/check-phone")
def check_phone(body: PhoneCheckRequest) -> dict[str, bool]:
    with Session(engine) as db:
        user = db.scalar(select(User).where(User.phone == body.phone))
        return {"exists": user is not None}


@router.post("/login")
def login(body: LoginRequest, response: Response) -> dict[str, Any]:
    with Session(engine) as db:
        user = db.scalar(select(User).where(User.phone == body.phone))
        if user is None or user.hashed_password is None:
            raise HTTPException(status_code=401, detail=AUTH_FAILURE_DETAIL)
        if not verify_password(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail=AUTH_FAILURE_DETAIL)
        tokens = issue_token_pair(user.id)
        _set_session_cookies(response, tokens)
        return _auth_response_payload(user, tokens)


def _normalize_invite_code(raw: str) -> str:
    return "".join(ch for ch in (raw or "").upper() if ch.isalnum())


def _generate_invite_code_value() -> str:
    # 6-char Crockford-ish random code (A-Z0-9), exclude confusing 0/O/1/I lightly by full alphabet ok
    return "".join(secrets.choice(_INVITE_ALPHABET) for _ in range(6))


def _user_can_manage_invites(user: User) -> bool:
    phone = (user.phone or "").strip()
    return bool(BOOTSTRAP_ADMIN_PHONE) and phone == BOOTSTRAP_ADMIN_PHONE


@router.post("/register")
def register(body: RegisterRequest, response: Response) -> dict[str, Any]:
    code_value = _normalize_invite_code(body.invite_code)
    if len(code_value) != 6:
        raise HTTPException(status_code=400, detail="请填写 6 位邀请码")

    with Session(engine) as db:
        existing = db.scalar(select(User).where(User.phone == body.phone))
        if existing is not None:
            raise HTTPException(status_code=409, detail="该手机号已注册")

        invite = db.scalar(select(InviteCode).where(InviteCode.code == code_value))
        now = datetime.utcnow()
        if invite is None:
            raise HTTPException(status_code=400, detail="邀请码无效")
        if invite.used_by is not None or invite.used_at is not None:
            raise HTTPException(status_code=400, detail="邀请码已被使用")
        if invite.expires_at <= now:
            raise HTTPException(status_code=400, detail="邀请码已过期")

        user = build_registration_user(body.phone, body.password)
        db.add(user)
        db.flush()  # allocate user.id
        invite.used_by = user.id
        invite.used_at = now
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="该手机号已注册") from exc
        db.refresh(user)
        tokens = issue_token_pair(user.id)
        _set_session_cookies(response, tokens)
        return _auth_response_payload(user, tokens)


@router.post("/invite-codes")
def create_invite_code(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Bootstrap admin only: mint a one-time invite code valid for N days."""
    if not _user_can_manage_invites(current_user):
        raise HTTPException(status_code=403, detail="无权生成邀请码")

    ttl_days = max(1, int(INVITE_CODE_TTL_DAYS))
    now = datetime.utcnow()
    expires_at = now + timedelta(days=ttl_days)

    with Session(engine) as db:
        code_value = None
        for _ in range(8):
            candidate = _generate_invite_code_value()
            exists = db.scalar(select(InviteCode.id).where(InviteCode.code == candidate))
            if exists is None:
                code_value = candidate
                break
        if not code_value:
            raise HTTPException(status_code=500, detail="邀请码生成失败，请重试")

        row = InviteCode(
            id=str(uuid.uuid4()),
            code=code_value,
            created_by=current_user.id,
            created_at=now,
            expires_at=expires_at,
        )
        db.add(row)
        db.commit()
        return {
            "code": code_value,
            "expires_at": expires_at.isoformat() + "Z",
            "ttl_days": ttl_days,
        }


@router.get("/invite-codes")
def list_invite_codes(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    if not _user_can_manage_invites(current_user):
        raise HTTPException(status_code=403, detail="无权查看邀请码")

    with Session(engine) as db:
        rows = db.scalars(
            select(InviteCode)
            .where(InviteCode.created_by == current_user.id)
            .order_by(InviteCode.created_at.desc())
            .limit(30)
        ).all()
        now = datetime.utcnow()
        items = []
        for row in rows:
            status = "active"
            if row.used_at is not None:
                status = "used"
            elif row.expires_at <= now:
                status = "expired"
            items.append(
                {
                    "code": row.code,
                    "status": status,
                    "created_at": row.created_at.isoformat() + "Z" if row.created_at else None,
                    "expires_at": row.expires_at.isoformat() + "Z" if row.expires_at else None,
                    "used_at": row.used_at.isoformat() + "Z" if row.used_at else None,
                }
            )
        return {"codes": items, "can_manage": True}


@router.get("/capabilities")
def auth_capabilities(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "can_manage_invites": _user_can_manage_invites(current_user),
    }


@router.post("/refresh")
def refresh_session(
    request: Request,
    response: Response,
    body: Optional[RefreshRequest] = None,
) -> dict[str, Any]:
    body_token = body.refresh_token if body is not None else None
    cookie_token = request.cookies.get(REFRESH_COOKIE_NAME)
    token = body_token or cookie_token
    if not token:
        raise HTTPException(status_code=401, detail="登录状态无效或已过期")

    # Cookie-based refresh is a browser session action and needs CSRF.
    if cookie_token and not body_token:
        validate_csrf(request)

    payload = decode_token(token, expected_type="refresh")
    user = _load_user(payload["sub"])
    tokens = issue_token_pair(user.id)
    _set_session_cookies(response, tokens)
    return _auth_response_payload(user, tokens)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    if request.cookies.get(REFRESH_COOKIE_NAME) or request.cookies.get("access_token"):
        validate_csrf(request)
    clear_auth_cookies(response)
    return {"message": "已退出登录"}


@router.get("/me")
def read_me(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user_id": current_user.id,
        "phone": current_user.phone,
        "username": current_user.username,
        "can_manage_invites": _user_can_manage_invites(current_user),
    }
