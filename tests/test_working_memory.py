"""Unit tests for near/overflow working memory."""

from __future__ import annotations

from services.working_memory import (
    _fallback_summary,
    format_turns_transcript,
    split_near_overflow,
    summarize_overflow_turns,
)
from routers.chat import build_system_prompt


def test_split_near_overflow_no_overlap_when_over_limit() -> None:
    turns = [{"role": "user", "content": f"u{i}", "id": i} for i in range(1, 11)]
    near, overflow = split_near_overflow(turns, near_limit=4)
    assert [t["id"] for t in near] == [7, 8, 9, 10]
    assert [t["id"] for t in overflow] == [1, 2, 3, 4, 5, 6]
    near_ids = {t["id"] for t in near}
    overflow_ids = {t["id"] for t in overflow}
    assert near_ids.isdisjoint(overflow_ids)


def test_split_near_overflow_empty_overflow_when_short() -> None:
    turns = [
        {"role": "user", "content": "a", "id": 1},
        {"role": "assistant", "content": "b", "id": 2},
    ]
    near, overflow = split_near_overflow(turns, near_limit=6)
    assert near == turns
    assert overflow == []


def test_fallback_summary_keeps_first_and_last_user() -> None:
    overflow = [
        {"role": "user", "content": "晚归怎么处理"},
        {"role": "assistant", "content": "扣分"},
        {"role": "user", "content": "夜不归宿呢"},
    ]
    text = _fallback_summary(overflow, max_chars=100)
    assert "晚归怎么处理" in text
    assert "夜不归宿呢" in text


def test_summarize_overflow_empty_returns_empty() -> None:
    assert summarize_overflow_turns([]) == ""


def test_format_transcript_truncates() -> None:
    turns = [{"role": "user", "content": "字" * 50}]
    text = format_turns_transcript(turns, max_chars=20)
    assert len(text) <= 20
    assert text.endswith("…")


def test_build_system_prompt_uses_overflow_not_near_duplicate() -> None:
    prompt = build_system_prompt(
        {
            "knowledge": "手册条文",
            "web": "无",
            "long_term": "",
            "overflow_summary": "早前讨论了晚归扣分",
            "resolved_query": "夜不归宿怎么处理",
        }
    )
    assert "【会话早前要点】" in prompt
    assert "早前讨论了晚归扣分" in prompt
    assert "【近期对话摘要】" not in prompt
    assert "【当前问题（含追问消解）】" in prompt
    # No long-term block when empty.
    assert "【长期记忆】" not in prompt


def test_build_system_prompt_omits_overflow_when_empty() -> None:
    prompt = build_system_prompt(
        {
            "knowledge": "手册条文",
            "web": "无",
            "overflow_summary": "",
        }
    )
    assert "【会话早前要点】\n" not in prompt
