"""On-demand summaries that are deliberately separate from long-term memory."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from services.llm import get_llm_client
from settings import MODEL_NAME


class ConversationSummary(BaseModel):
    topic: str = Field(min_length=1)
    confirmed_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


def summarize_conversation(messages: list[dict[str, Any]]) -> ConversationSummary:
    transcript = "\n".join(
        f"[{item.get('role', 'unknown')}]: {item.get('content', '')}"
        for item in messages
        if str(item.get("content", "")).strip()
    )
    prompt = (
        "请总结以下会话，仅输出 JSON 对象，不要 Markdown。\n"
        '字段固定为 topic、confirmed_points、open_questions、next_actions，后三项都是字符串数组。\n\n'
        f"{transcript}"
    )
    response = get_llm_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        return ConversationSummary.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError):
        return ConversationSummary(
            topic="当前会话",
            confirmed_points=[content] if content else [],
        )
