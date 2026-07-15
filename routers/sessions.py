"""Session management routes."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import asc, desc, or_, select
from sqlalchemy.orm import Session

from database import engine
from models import ChatSession, Message, User
from security import get_current_user

router = APIRouter(tags=["sessions"])

DEFAULT_SESSION_TITLE = "新建对话"
SESSION_TITLE_LIMIT = 18
SESSION_PREVIEW_LIMIT = 60
SEARCH_SNIPPET_LIMIT = 80
SEARCH_RESULT_LIMIT = 30
SESSION_TITLE_MAX_LENGTH = 100


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


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=SESSION_TITLE_MAX_LENGTH)


@router.patch("/sessions/{session_id}")
def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    new_title = " ".join(body.title.strip().split())
    if not new_title:
        raise HTTPException(status_code=422, detail="标题不能为空。")
    if len(new_title) > SESSION_TITLE_MAX_LENGTH:
        new_title = new_title[:SESSION_TITLE_MAX_LENGTH]

    with Session(engine) as db:
        chat_window = db.scalar(select(ChatSession).where(ChatSession.id == session_id))
        if chat_window is None:
            raise HTTPException(status_code=404, detail="会话不存在。")
        if chat_window.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权修改该会话。")
        chat_window.title = new_title
        db.commit()
        return {
            "session_id": chat_window.id,
            "title": chat_window.title,
        }


def _build_search_snippet(content: str, keyword: str) -> str:
    compact = " ".join((content or "").strip().split())
    if not compact:
        return ""
    lower_content = compact.lower()
    lower_keyword = keyword.lower()
    idx = lower_content.find(lower_keyword)
    if idx < 0:
        return compact[:SEARCH_SNIPPET_LIMIT] + ("..." if len(compact) > SEARCH_SNIPPET_LIMIT else "")
    half = SEARCH_SNIPPET_LIMIT // 2
    start = max(0, idx - half)
    end = min(len(compact), idx + len(keyword) + half)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return prefix + compact[start:end] + suffix


@router.get("/sessions/search")
def search_sessions(
    q: str = Query(..., min_length=1, description="关键词，匹配会话标题或消息内容"),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    keyword = q.strip()
    if not keyword:
        return {"query": q, "results": []}

    like_pattern = f"%{keyword}%"

    with Session(engine) as db:
        sessions = db.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == current_user.id)
            .where(
                or_(
                    ChatSession.title.ilike(like_pattern),
                    ChatSession.id.in_(
                        select(Message.session_id).where(Message.content.ilike(like_pattern))
                    ),
                )
            )
            .order_by(desc(ChatSession.id))
            .limit(SEARCH_RESULT_LIMIT)
        ).all()

        if not sessions:
            return {"query": q, "results": []}

        session_ids = [s.id for s in sessions]
        messages_by_session: dict[str, list[Message]] = {}
        for msg in db.scalars(
            select(Message)
            .where(Message.session_id.in_(session_ids))
            .order_by(asc(Message.id))
        ).all():
            messages_by_session.setdefault(msg.session_id, []).append(msg)

    results: list[dict[str, Any]] = []
    for session in sessions:
        title_hit = keyword.lower() in (session.title or "").lower()
        first_msg_content = ""
        matched_message: Optional[str] = None
        msgs = messages_by_session.get(session.id, [])
        if msgs:
            first_msg_content = msgs[0].content or ""

        for msg in msgs:
            if keyword.lower() in (msg.content or "").lower():
                matched_message = _build_search_snippet(msg.content, keyword)
                break

        results.append(
            {
                "session_id": session.id,
                "title": session.title or DEFAULT_SESSION_TITLE,
                "preview": matched_message or _build_search_snippet(first_msg_content, keyword),
                "matched_in_title": title_hit,
                "matched_in_message": matched_message is not None,
                "message_count": len(msgs),
            }
        )

    return {"query": q, "results": results}
