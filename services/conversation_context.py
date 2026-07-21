"""Multi-turn conversation helpers: follow-up detection and query resolution."""

from __future__ import annotations

import re
from typing import Any

# Short / deictic follow-ups that usually need the previous user turn.
_FOLLOWUP_PREFIX_RE = re.compile(
    r"^(那|那么|然后|还有|另外|以及|如果|要是|假如|若|那如果|那要是|再问|接着)"
)
_FOLLOWUP_MARKERS = (
    "呢",
    "吗",
    "嘛",
    "吧",
    "这个",
    "那个",
    "这样",
    "那样",
    "上述",
    "上面",
    "刚才",
    "前面",
    "同上",
    "继续",
)
_STANDALONE_TOPIC_HINTS = (
    "奖学金",
    "助学金",
    "综测",
    "转学",
    "转专业",
    "休学",
    "复学",
    "退学",
    "旷课",
    "晚归",
    "夜不归宿",
    "宿舍",
    "寝室",
    "处分",
    "请假",
    "毕业",
    "学位",
    "注册",
    "报到",
    "补考",
    "重修",
    "天气",
    "课表",
    "图书馆",
)


def is_followup_utterance(current: str, previous_user: str | None = None) -> bool:
    """Heuristic: short/deictic replies that depend on prior user question."""
    text = (current or "").strip()
    if not text:
        return False
    prev = (previous_user or "").strip()
    if not prev:
        return False

    # Explicit continuation / conditional tails.
    if _FOLLOWUP_PREFIX_RE.search(text):
        return True
    if any(marker in text for marker in _FOLLOWUP_MARKERS) and len(text) <= 24:
        return True
    # Very short replies without a clear new topic.
    if len(text) <= 12 and not any(hint in text for hint in _STANDALONE_TOPIC_HINTS):
        return True
    # Conditional fragment without main topic words.
    if text.startswith(("如果", "要是", "假如", "若")) and not any(
        hint in text for hint in _STANDALONE_TOPIC_HINTS
    ):
        return True
    return False


def resolve_followup_query(current: str, previous_user: str | None) -> str:
    """Build a retrieval query that keeps the prior topic for follow-ups."""
    text = (current or "").strip()
    prev = (previous_user or "").strip()
    if not text:
        return prev
    if not prev or not is_followup_utterance(text, prev):
        return text
    # Keep both sides: original topic + current constraint.
    return f"{prev}\n用户追问：{text}"


def format_turns_for_prompt(turns: list[dict[str, str]], *, limit: int = 6) -> str:
    """Render recent turns as plain text for system-side short-term memory."""
    if not turns:
        return ""
    lines: list[str] = []
    for turn in turns[-limit:]:
        role = (turn.get("role") or "").strip() or "user"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"[{label}] {content}")
    return "\n".join(lines)


def turns_to_chat_messages(turns: list[dict[str, str]], *, exclude_last_user: bool = True) -> list[dict[str, str]]:
    """Convert stored turns into OpenAI-style chat messages.

    When exclude_last_user=True, drop the trailing user turn if it equals the
    current request (caller will append the live user message separately).
    """
    items = list(turns or [])
    if exclude_last_user and items and items[-1].get("role") == "user":
        items = items[:-1]
    messages: list[dict[str, str]] = []
    for turn in items:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def last_user_utterance(turns: list[dict[str, str]]) -> str | None:
    for turn in reversed(turns or []):
        if turn.get("role") == "user":
            content = (turn.get("content") or "").strip()
            if content:
                return content
    return None


def previous_user_utterance(turns: list[dict[str, str]], current: str) -> str | None:
    """Previous user turn before the current one (if current is already saved)."""
    text = (current or "").strip()
    seen_current = False
    for turn in reversed(turns or []):
        if turn.get("role") != "user":
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if not seen_current and content == text:
            seen_current = True
            continue
        return content
    return None
