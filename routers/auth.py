"""Authentication routes."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import engine
from models import User
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

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_FAILURE_DETAIL = "手机号或密码错误"


class PhoneCheckRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)


class LoginRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=32)


class RegisterRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=32)


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


@router.post("/register")
def register(body: RegisterRequest, response: Response) -> dict[str, Any]:
    with Session(engine) as db:
        existing = db.scalar(select(User).where(User.phone == body.phone))
        if existing is not None:
            raise HTTPException(status_code=409, detail="该手机号已注册")
        user = build_registration_user(body.phone, body.password)
        db.add(user)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="该手机号已注册") from exc
        db.refresh(user)
        tokens = issue_token_pair(user.id)
        _set_session_cookies(response, tokens)
        return _auth_response_payload(user, tokens)


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
    }
