"""Structured, auditable memory that only stores explicit user statements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import AgentMemory, MemoryEvent


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    memory_key: str
    memory_value: str
    confidence: float


class MemoryManager:
    def extract_explicit_candidates(self, content: str) -> list[MemoryCandidate]:
        text = " ".join((content or "").strip().split())
        candidates: list[MemoryCandidate] = []
        style_match = re.search(r"(?:以后|请).{0,8}(?:回答|回复).{0,4}(简洁|详细|分点|直接)", text)
        if style_match:
            candidates.append(MemoryCandidate("preference", "answer_style", style_match.group(1), 0.95))

        major_match = re.search(r"我是(.{2,20}?)专业", text)
        if major_match:
            candidates.append(MemoryCandidate("profile", "major", major_match.group(1), 0.95))
        return candidates

    def upsert(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: Optional[str],
        memory_type: str,
        memory_key: str,
        memory_value: str,
        confidence: float,
        source_message_id: Optional[int] = None,
    ) -> AgentMemory:
        active = db.scalars(
            select(AgentMemory).where(
                AgentMemory.user_id == user_id,
                AgentMemory.memory_type == memory_type,
                AgentMemory.memory_key == memory_key,
                AgentMemory.status == "active",
            )
        ).all()
        for old in active:
            old.status = "superseded"
            db.add(MemoryEvent(
                memory_id=old.id,
                event_type="superseded",
                old_value=old.memory_value,
                new_value=memory_value,
                reason="同一记忆键被用户明确更新",
                source_message_id=source_message_id,
            ))

        memory = AgentMemory(
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            memory_key=memory_key,
            memory_value=memory_value,
            confidence=confidence,
            status="active",
            source_message_id=source_message_id,
        )
        db.add(memory)
        db.flush()
        db.add(MemoryEvent(
            memory_id=memory.id,
            event_type="created",
            old_value=None,
            new_value=memory_value,
            reason="用户明确表达的稳定信息",
            source_message_id=source_message_id,
        ))
        return memory

    def store_explicit_candidates(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: Optional[str],
        content: str,
        source_message_id: Optional[int] = None,
    ) -> list[AgentMemory]:
        return [
            self.upsert(
                db,
                user_id=user_id,
                session_id=session_id,
                memory_type=candidate.memory_type,
                memory_key=candidate.memory_key,
                memory_value=candidate.memory_value,
                confidence=candidate.confidence,
                source_message_id=source_message_id,
            )
            for candidate in self.extract_explicit_candidates(content)
        ]

    def render_active_context(self, db: Session, user_id: str) -> str:
        memories = db.scalars(
            select(AgentMemory)
            .where(AgentMemory.user_id == user_id, AgentMemory.status == "active")
            .order_by(AgentMemory.updated_at.desc())
            .limit(12)
        ).all()
        return "\n".join(
            f"[{memory.memory_type}] {memory.memory_key}: {memory.memory_value}"
            for memory in memories
        )

    def audit_event_count(self, db: Session, memory_id: int) -> int:
        return int(db.scalar(select(func.count()).select_from(MemoryEvent).where(MemoryEvent.memory_id == memory_id)) or 0)
