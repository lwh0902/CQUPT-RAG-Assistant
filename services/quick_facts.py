"""Quick facts: human-reviewed direct answers for a small set of high-frequency asks.

Design rules:
- Only facts with verified=true ever match. Unverified stubs are invisible.
- Answers come from the curated JSON, never from the model — a wrong phone
  number is worse than no answer.
- Matching is conservative regex on the raw user query; no RAG, no LLM call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from settings import BASE_DIR

QUICK_FACTS_PATH = BASE_DIR / "quick_facts.json"


@dataclass(frozen=True)
class QuickFact:
    id: str
    title: str
    answer: str
    source_name: str
    source_url: str
    updated_at: str
    sample_question: str
    patterns: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "answer": self.answer,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "updated_at": self.updated_at,
        }


_cache: dict[str, Any] = {"mtime": None, "facts": []}


def load_quick_facts(path: Path | None = None) -> list[QuickFact]:
    """Load verified facts only, cached by file mtime."""
    target = path or QUICK_FACTS_PATH
    if not target.exists():
        return []
    mtime = target.stat().st_mtime
    if path is None and _cache["mtime"] == mtime:
        return _cache["facts"]

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    facts: list[QuickFact] = []
    for item in payload.get("facts", []):
        if not item.get("verified"):
            continue
        patterns = tuple(p for p in item.get("patterns", []) if p)
        if not patterns:
            continue
        facts.append(
            QuickFact(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or ""),
                answer=str(item.get("answer") or ""),
                source_name=str(item.get("source_name") or ""),
                source_url=str(item.get("source_url") or ""),
                updated_at=str(item.get("updated_at") or ""),
                sample_question=str(item.get("sample_question") or item.get("title") or ""),
                patterns=patterns,
            )
        )
    if path is None:
        _cache["mtime"] = mtime
        _cache["facts"] = facts
    return facts


def match_quick_fact(query: str, *, path: Path | None = None) -> QuickFact | None:
    """Return the first verified fact whose pattern matches the query."""
    text = " ".join((query or "").strip().split())
    if not text or len(text) > 80:
        # Long inputs are real questions, not quick-fact lookups.
        return None
    for fact in load_quick_facts(path):
        for pattern in fact.patterns:
            try:
                if re.search(pattern, text):
                    return fact
            except re.error:
                continue
    return None


def list_public_facts(*, path: Path | None = None) -> list[dict[str, Any]]:
    """Verified facts for the empty-state quick links (no answers exposed here)."""
    return [
        {"id": fact.id, "title": fact.title, "sample_question": fact.sample_question}
        for fact in load_quick_facts(path)
    ]


def render_quick_fact_reply(fact: QuickFact) -> str:
    source_line = fact.source_name
    if fact.updated_at:
        source_line = f"{source_line} · {fact.updated_at} 更新"
    return f"**{fact.title}**\n\n{fact.answer}\n\n来源：{source_line}"
