"""Session management routes."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session

from database import engine
from models import ChatSession, Message, User
from security import get_current_user

router = APIRouter(tags=["sessions"])

DEFAULT_SESSION_TITLE = "新建对话"
SESSION_TITLE_LIMIT = 18
SESSION_PREVIEW_LIMIT = 60


def normalize_message_role(role: str) -> str:
    return "assistant" if role in {"assistant", "ai"} else role


def build_session_title(message: str) -> str:
    compact_message = " ".join((message or "").strip().split())
    if not compact_message:
        return DEFAULT_SESSION_TITLE
    if len(compact_message) <= SESSION_TITLE_LIMIT:
        return compact_message
    return f"{compact_message[:SESSION_TITLE_LIMIT]}..."


def build_message_preview(message: str) -> str:
    compact_message = " ".join((message or "").strip().split())
    if len(compact_message) <= SESSION_PREVIEW_LIMIT:
        return compact_message
    return f"{compact_message[:SESSION_PREVIEW_LIMIT]}..."


def serialize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            "id": message.id,
            "role": normalize_message_role(message.role),
            "content": message.content,
        }
        for message in messages
    ]


def list_user_sessions(
    user_id: str,
    cursor: Optional[int] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    with Session(engine) as db:
        query = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.id))
        )
        if cursor is not None:
            query = query.where(ChatSession.id < str(cursor))
        sessions = db.scalars(query.limit(limit)).all()

        if not sessions:
            return []

        session_ids = [session.id for session in sessions]
        session_meta = {
            session.id: {
                "message_count": 0,
                "last_message_id": 0,
                "preview": "",
            }
            for session in sessions
        }

        messages = db.scalars(
            select(Message)
            .where(Message.session_id.in_(session_ids))
            .order_by(desc(Message.id))
        ).all()

        for message in messages:
            meta = session_meta.get(message.session_id)
            if meta is None:
                continue
            meta["message_count"] += 1
            if meta["last_message_id"] == 0:
                meta["last_message_id"] = message.id
                meta["preview"] = build_message_preview(message.content)

    summaries = [
        {
            "session_id": session.id,
            "title": session.title or DEFAULT_SESSION_TITLE,
            "preview": session_meta[session.id]["preview"],
            "message_count": session_meta[session.id]["message_count"],
        }
        for session in sessions
    ]
    return summaries


def get_session_messages(user_id: str, session_id: str) -> list[dict[str, Any]]:
    with Session(engine) as db:
        chat_window = db.scalar(select(ChatSession).where(ChatSession.id == session_id))
        if chat_window is None:
            raise HTTPException(status_code=404, detail="会话不存在。")
        if chat_window.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问该聊天会话。")

        messages = db.scalars(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(asc(Message.id))
        ).all()

    return serialize_messages(messages)


@router.get("/sessions")
def read_sessions(
    cursor: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> dict[str, list[dict[str, Any]]]:
    return {"sessions": list_user_sessions(current_user.id, cursor=cursor, limit=limit)}


@router.get("/sessions/{session_id}/messages")
def read_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "messages": get_session_messages(current_user.id, session_id),
    }


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    with Session(engine) as db:
        chat_window = db.scalar(select(ChatSession).where(ChatSession.id == session_id))
        if chat_window is None:
            raise HTTPException(status_code=404, detail="会话不存在。")
        if chat_window.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权删除该会话。")
        db.delete(chat_window)
        db.commit()
    return {"message": "已删除"}
