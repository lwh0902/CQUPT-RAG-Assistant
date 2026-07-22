"""Session working memory: near-window turns + single rolling overflow summary in MySQL."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import ChatSession
from services.llm import get_llm_client
from settings import (
    MODEL_NAME,
    OVERFLOW_SUMMARY_ENABLED,
    OVERFLOW_SUMMARY_MAX_CHARS,
    SHORT_TERM_MESSAGE_LIMIT,
)

logger = logging.getLogger(__name__)


def split_near_overflow(
    turns: list[dict[str, Any]],
    *,
    near_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split chronological turns into (near_window, overflow_prefix).

    near and overflow never overlap. Current live user turn should be excluded
    by the caller or stripped via turns_to_chat_messages(exclude_last_user=True).
    """
    limit = SHORT_TERM_MESSAGE_LIMIT if near_limit is None else max(0, int(near_limit))
    items = list(turns or [])
    if limit <= 0 or len(items) <= limit:
        return items, []
    return items[-limit:], items[:-limit]


def format_turns_transcript(turns: list[dict[str, Any]], *, max_chars: int = 4000) -> str:
    lines: list[str] = []
    for turn in turns:
        role = (turn.get("role") or "").strip() or "user"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"[{label}] {content}")
    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _fallback_summary(overflow: list[dict[str, Any]], *, max_chars: int) -> str:
    """Rule fallback when LLM summary is unavailable."""
    users = [
        (turn.get("content") or "").strip()
        for turn in overflow
        if turn.get("role") == "user" and (turn.get("content") or "").strip()
    ]
    if not users:
        return ""
    # Keep first topic + last user ask from overflow.
    parts = [users[0]]
    if len(users) > 1:
        parts.append(users[-1])
    text = "；".join(parts)
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def summarize_overflow_turns(
    overflow: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
    previous_summary: str | None = None,
) -> str:
    """Collapse overflow turns into one short Chinese summary for the system prompt."""
    if not overflow:
        return ""
    limit = OVERFLOW_SUMMARY_MAX_CHARS if max_chars is None else max_chars
    transcript = format_turns_transcript(overflow)
    if not transcript:
        return ""

    prev = (previous_summary or "").strip()
    prompt = (
        "请把以下更早的对话压缩成一段中文「会话早前要点」，供助手继续对话使用。\n"
        f"要求：不超过{limit}字；只总结对话里出现的主题、已确认事实/条件、未决问题；\n"
        "不要补充政策条文，不要编造手册内容，不要输出 Markdown 标题。\n"
        "若几乎无信息，只输出空字符串。\n\n"
    )
    if prev:
        prompt += f"【上一版早前要点】\n{prev}\n\n"
    prompt += f"【需压缩的更早对话】\n{transcript}\n\n【会话早前要点】"

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            extra_body={"thinking": {"type": "disabled"}},
        )
        summary = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("Overflow summary LLM failed; using rule fallback", exc_info=True)
        return _fallback_summary(overflow, max_chars=limit)

    if not summary or summary in {"无", "（无）", "无信息", "空"}:
        return _fallback_summary(overflow, max_chars=limit)
    if len(summary) > limit:
        summary = summary[: limit - 1].rstrip() + "…"
    return summary


def _overflow_bounds(overflow: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    ids: list[int] = []
    for turn in overflow:
        raw = turn.get("id")
        try:
            if raw is not None:
                ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not ids:
        return None, None
    return min(ids), max(ids)


def load_or_refresh_overflow_summary(
    db: Session,
    *,
    session_id: str,
    overflow: list[dict[str, Any]],
    enabled: bool | None = None,
) -> str:
    """Return one rolling overflow summary; persist on ChatSession in MySQL.

    Cache key: overflow_until_message_id (last overflow message id).
    Only one summary row/fields per session — later refreshes overwrite.
    """
    active = OVERFLOW_SUMMARY_ENABLED if enabled is None else enabled
    session = db.get(ChatSession, session_id)
    if session is None:
        return ""

    if not active or not overflow:
        if session.overflow_summary or session.overflow_until_message_id is not None:
            session.overflow_summary = None
            session.overflow_until_message_id = None
            session.overflow_from_message_id = None
            session.overflow_updated_at = datetime.utcnow()
            db.add(session)
            db.commit()
        return ""

    from_id, until_id = _overflow_bounds(overflow)
    if (
        until_id is not None
        and session.overflow_until_message_id == until_id
        and (session.overflow_summary or "").strip()
    ):
        return (session.overflow_summary or "").strip()

    summary = summarize_overflow_turns(
        overflow,
        previous_summary=session.overflow_summary,
    )
    session.overflow_summary = summary or None
    session.overflow_until_message_id = until_id
    session.overflow_from_message_id = from_id
    session.overflow_updated_at = datetime.utcnow()
    db.add(session)
    db.commit()
    return summary
