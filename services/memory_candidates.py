"""Validate model-proposed memories before users can confirm them."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class MemoryCandidatePayload(BaseModel):
    memory_type: Literal["preference", "profile", "goal", "constraint"]
    memory_key: str = Field(min_length=1, max_length=64)
    memory_value: str = Field(min_length=1, max_length=240)
    reason: str = Field(min_length=1, max_length=160)


class MemoryCandidateEnvelope(BaseModel):
    candidates: list[MemoryCandidatePayload] = Field(default_factory=list, max_length=2)


_SENSITIVE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"\b(?:sk-|Bearer\s+|api[_-]?key|token|password)\S+", re.I),
)
_SENSITIVE_KEYS = {"phone", "password", "token", "api_key", "address", "medical", "finance"}


def parse_memory_candidates(raw: str) -> list[MemoryCandidatePayload]:
    try:
        envelope = MemoryCandidateEnvelope.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return []
    return [item for item in envelope.candidates if _is_safe(item)]


def _is_safe(candidate: MemoryCandidatePayload) -> bool:
    combined = f"{candidate.memory_key} {candidate.memory_value}".lower()
    return candidate.memory_key.lower() not in _SENSITIVE_KEYS and not any(pattern.search(combined) for pattern in _SENSITIVE_PATTERNS)
