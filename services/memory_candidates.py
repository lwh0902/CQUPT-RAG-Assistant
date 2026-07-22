"""Validate and extract model-proposed memories before they can affect context."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from settings import (
    MEMORY_AUTO_CONFIRM_MIN_CONFIDENCE,
    MEMORY_LLM_EXTRACT_ENABLED,
    MEMORY_PENDING_MIN_CONFIDENCE,
)

logger = logging.getLogger(__name__)

MemoryType = Literal["preference", "profile", "goal", "constraint"]
GateDecision = Literal["auto", "pending", "reject"]

# Only these keys may become durable memory (production whitelist).
ALLOWED_MEMORY_KEYS: dict[str, set[str]] = {
    "preference": {"answer_style", "language"},
    "profile": {"major", "college", "grade"},
    "goal": {"current_task"},
    "constraint": {"no_schedule", "no_weather", "session_topic"},
}

_SENSITIVE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"\b(?:sk-|Bearer\s+|api[_-]?key|token|password)\S+", re.I),
    re.compile(r"(?<!\d)\d{8,12}(?!\d)"),  # student-id-ish runs
)
_SENSITIVE_KEYS = {
    "phone",
    "password",
    "token",
    "api_key",
    "address",
    "medical",
    "finance",
    "id_card",
    "student_id",
    "学号",
    "手机",
}
_EMOTION_HINTS = ("烦", "开心", "难过", "郁闷", "无聊", "哈哈", "嘿嘿", "心情")
_STABLE_HINTS = (
    "我是",
    "我在",
    "专业",
    "学院",
    "年级",
    "请",
    "以后",
    "记住",
    "偏好",
    "简洁",
    "详细",
    "分点",
    "不要推荐",
    "别推荐",
    "不要查",
)


class MemoryCandidatePayload(BaseModel):
    memory_type: MemoryType
    memory_key: str = Field(min_length=1, max_length=64)
    memory_value: str = Field(min_length=1, max_length=240)
    reason: str = Field(default="user_stated", min_length=1, max_length=160)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    explicit: bool = True

    @field_validator("memory_key")
    @classmethod
    def _normalize_key(cls, value: str) -> str:
        return (value or "").strip().lower().replace(" ", "_")

    @field_validator("memory_value")
    @classmethod
    def _normalize_value(cls, value: str) -> str:
        return " ".join((value or "").strip().split())


class MemoryCandidateEnvelope(BaseModel):
    candidates: list[MemoryCandidatePayload] = Field(default_factory=list, max_length=2)


@dataclass(frozen=True)
class GatedMemoryCandidate:
    payload: MemoryCandidatePayload
    decision: GateDecision
    reject_reason: str = ""


@dataclass(frozen=True)
class MemoryAction:
    """Client-facing memory side-effect for toast / confirm UI."""

    action: Literal["saved", "pending", "rejected"]
    memory_type: str
    memory_key: str
    memory_value: str
    candidate_id: Optional[str] = None
    memory_id: Optional[int] = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "memory_type": self.memory_type,
            "memory_key": self.memory_key,
            "memory_value": self.memory_value,
            "candidate_id": self.candidate_id,
            "memory_id": self.memory_id,
            "message": self.message,
        }


def looks_like_stable_info(content: str) -> bool:
    """Cheap prefilter: skip LLM when the utterance is unlikely to carry profile data."""
    text = " ".join((content or "").strip().split())
    if not text or len(text) < 2:
        return False
    if any(hint in text for hint in _EMOTION_HINTS) and not any(
        hint in text for hint in _STABLE_HINTS
    ):
        return False
    if any(hint in text for hint in _STABLE_HINTS):
        return True
    # Explicit remember phrasing.
    if "记住" in text or "别忘了" in text:
        return True
    return False


def parse_memory_candidates(raw: str) -> list[MemoryCandidatePayload]:
    """Parse model JSON and keep only structurally valid + non-sensitive items.

    Gate tier (auto/pending/reject) is applied separately via ``gate_candidate``.
    """
    text = (raw or "").strip()
    if not text:
        return []
    # Tolerate fenced JSON from the model.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        envelope = MemoryCandidateEnvelope.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return []
    safe: list[MemoryCandidatePayload] = []
    for item in envelope.candidates:
        if _is_sensitive(item):
            continue
        if not _is_whitelisted(item):
            continue
        safe.append(item)
    return safe[:2]


def gate_candidate(candidate: MemoryCandidatePayload) -> GatedMemoryCandidate:
    """Three-tier write gate: auto / pending / reject."""
    if _is_sensitive(candidate):
        return GatedMemoryCandidate(candidate, "reject", "sensitive")
    if not _is_whitelisted(candidate):
        return GatedMemoryCandidate(candidate, "reject", "not_whitelisted")
    if not (candidate.memory_value or "").strip():
        return GatedMemoryCandidate(candidate, "reject", "empty_value")

    conf = float(candidate.confidence or 0.0)
    if candidate.explicit and conf >= MEMORY_AUTO_CONFIRM_MIN_CONFIDENCE:
        return GatedMemoryCandidate(candidate, "auto")
    if conf >= MEMORY_PENDING_MIN_CONFIDENCE:
        return GatedMemoryCandidate(candidate, "pending")
    return GatedMemoryCandidate(candidate, "reject", "low_confidence")


def extract_regex_candidates(content: str) -> list[MemoryCandidatePayload]:
    """High-precision regex path (always-on, no LLM)."""
    text = " ".join((content or "").strip().split())
    out: list[MemoryCandidatePayload] = []

    # Accept both 「请简洁回答」 and 「回答请简洁」 orderings.
    style_match = re.search(
        r"(?:以后|请|希望你).{0,12}(?:回答|回复).{0,6}(简洁|详细|分点|直接)"
        r"|(?:以后|请|希望你).{0,12}(简洁|详细|分点|直接).{0,6}(?:回答|回复)",
        text,
    )
    if style_match:
        style_value = style_match.group(1) or style_match.group(2)
        if style_value:
            out.append(
                MemoryCandidatePayload(
                    memory_type="preference",
                    memory_key="answer_style",
                    memory_value=style_value,
                    reason="regex_answer_style",
                    confidence=0.95,
                    explicit=True,
                )
            )

    major_match = re.search(r"我是(.{2,20}?)专业", text)
    if major_match:
        out.append(
            MemoryCandidatePayload(
                memory_type="profile",
                memory_key="major",
                memory_value=major_match.group(1),
                reason="regex_major",
                confidence=0.95,
                explicit=True,
            )
        )

    college_match = re.search(r"我(?:是|在)(.{2,20}?)学院", text)
    if college_match:
        out.append(
            MemoryCandidatePayload(
                memory_type="profile",
                memory_key="college",
                memory_value=college_match.group(1) + "学院"
                if not college_match.group(1).endswith("学院")
                else college_match.group(1),
                reason="regex_college",
                confidence=0.92,
                explicit=True,
            )
        )

    grade_match = re.search(r"(?:我是|年级)?\s*(大[一二三四]|研[一二三]|[1-4]年级)", text)
    if grade_match and ("我" in text or "年级" in text):
        out.append(
            MemoryCandidatePayload(
                memory_type="profile",
                memory_key="grade",
                memory_value=grade_match.group(1),
                reason="regex_grade",
                confidence=0.9,
                explicit=True,
            )
        )

    if re.search(r"(?:不要|别|无需).{0,6}(?:推荐|查询|查)?课表", text):
        out.append(
            MemoryCandidatePayload(
                memory_type="constraint",
                memory_key="no_schedule",
                memory_value="true",
                reason="regex_no_schedule",
                confidence=0.93,
                explicit=True,
            )
        )

    return out


def extract_llm_candidates(content: str) -> list[MemoryCandidatePayload]:
    """Ask the chat model for at most two whitelist memory candidates."""
    if not MEMORY_LLM_EXTRACT_ENABLED:
        return []
    if not looks_like_stable_info(content):
        return []

    prompt = (
        "从用户这句话中抽取最多 2 条可长期保存的稳定信息。"
        "只允许以下 type/key：\n"
        "- preference/answer_style, preference/language\n"
        "- profile/major, profile/college, profile/grade\n"
        "- goal/current_task\n"
        "- constraint/no_schedule, constraint/no_weather, constraint/session_topic\n"
        "禁止：情绪、手机号、学号、密码、推断性标签、制度条文、一次性问题。\n"
        "若没有值得保存的信息，返回 {\"candidates\":[]}。\n"
        "只输出 JSON，字段：memory_type, memory_key, memory_value, reason, confidence(0-1), explicit(bool)。\n\n"
        f"用户：{content.strip()}\n"
    )
    try:
        from services.llm import get_llm_client
        from settings import MODEL_NAME

        response = get_llm_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("LLM memory extract failed", exc_info=True)
        return []
    return parse_memory_candidates(raw)


def collect_candidates(content: str, *, use_llm: bool | None = None) -> list[MemoryCandidatePayload]:
    """Merge regex + optional LLM candidates; de-dupe by type/key keeping higher confidence."""
    merged: dict[tuple[str, str], MemoryCandidatePayload] = {}
    for item in extract_regex_candidates(content):
        merged[(item.memory_type, item.memory_key)] = item

    enable_llm = MEMORY_LLM_EXTRACT_ENABLED if use_llm is None else use_llm
    if enable_llm:
        for item in extract_llm_candidates(content):
            key = (item.memory_type, item.memory_key)
            prev = merged.get(key)
            if prev is None or float(item.confidence) > float(prev.confidence):
                merged[key] = item
    return list(merged.values())[:2]


def format_memory_toast(action: str, memory_key: str, memory_value: str) -> str:
    label = {
        "answer_style": "回答风格",
        "language": "语言偏好",
        "major": "专业",
        "college": "学院",
        "grade": "年级",
        "current_task": "当前任务",
        "no_schedule": "不推荐课表",
        "no_weather": "不查天气",
        "session_topic": "会话主题",
    }.get(memory_key, memory_key)
    if action == "saved":
        return f"已记住：{label}={memory_value}"
    if action == "pending":
        return f"是否记住：{label}={memory_value}？"
    return f"未保存记忆：{label}"


def _is_whitelisted(candidate: MemoryCandidatePayload) -> bool:
    allowed = ALLOWED_MEMORY_KEYS.get(candidate.memory_type)
    if not allowed:
        return False
    return candidate.memory_key in allowed


def _is_sensitive(candidate: MemoryCandidatePayload) -> bool:
    key = (candidate.memory_key or "").lower()
    value = candidate.memory_value or ""
    combined = f"{key} {value}".lower()
    if key in _SENSITIVE_KEYS:
        return True
    if any(token in combined for token in _SENSITIVE_KEYS):
        return True
    return any(pattern.search(combined) for pattern in _SENSITIVE_PATTERNS)
