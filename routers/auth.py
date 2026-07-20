"""Authentication routes."""

import uuid

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import engine
from models import User
from security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class PhoneCheckRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)


class LoginRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=32)


class RegisterRequest(BaseModel):
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=32)


def build_registration_user(phone: str, password: str) -> User:
    """Create a user with an identifier that cannot collide on phone suffixes."""
    user_id = str(uuid.uuid4())
    return User(
        id=user_id,
        username=f"user_{user_id.replace('-', '')}",
        phone=phone,
        hashed_password=hash_password(password),
    )


@router.post("/check-phone")
def check_phone(body: PhoneCheckRequest) -> dict[str, bool]:
    with Session(engine) as db:
        user = db.scalar(select(User).where(User.phone == body.phone))
        return {"exists": user is not None}


@router.post("/login")
def login(body: LoginRequest) -> dict[str, str]:
    with Session(engine) as db:
        user = db.scalar(select(User).where(User.phone == body.phone))
        if user is None or user.hashed_password is None:
            raise HTTPException(status_code=401, detail="账号不存在")
        if not verify_password(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="密码错误")
        token = create_access_token(user.id)
        return {"token": token, "user_id": user.id, "phone": user.phone}


@router.post("/register")
def register(body: RegisterRequest) -> dict[str, str]:
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
        token = create_access_token(user.id)
        return {"token": token, "user_id": user.id, "phone": user.phone}


@router.get("/me")
def read_me(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user_id": current_user.id,
        "phone": current_user.phone,
        "username": current_user.username,
    }
