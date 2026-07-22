"""Structured, auditable memory with gated write path (auto / pending / reject)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import AgentMemory, MemoryCandidate as MemoryCandidateRow, MemoryEvent
from services.memory_candidates import (
    GatedMemoryCandidate,
    MemoryAction,
    MemoryCandidatePayload,
    collect_candidates,
    format_memory_toast,
    gate_candidate,
)
from settings import MEMORY_CANDIDATE_TTL_HOURS
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Lightweight aliases so Chinese queries can rank English memory keys.
_MEMORY_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "major": ("专业", "专业课", "学科"),
    "answer_style": ("简洁", "详细", "分点", "直接", "回答风格", "回复风格"),
    "college": ("学院", "院系"),
    "grade": ("年级", "大一", "大二", "大三", "大四"),
}


@dataclass(frozen=True)
class MemoryCandidate:
    """Backward-compatible alias used by older tests/callers."""

    memory_type: str
    memory_key: str
    memory_value: str
    confidence: float


class MemoryManager:
    def extract_explicit_candidates(self, content: str) -> list[MemoryCandidate]:
        """Legacy regex-only API kept for unit tests."""
        from services.memory_candidates import extract_regex_candidates

        return [
            MemoryCandidate(
                memory_type=item.memory_type,
                memory_key=item.memory_key,
                memory_value=item.memory_value,
                confidence=item.confidence,
            )
            for item in extract_regex_candidates(content)
        ]

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
        reason: str = "用户明确表达的稳定信息",
        confirmed: bool = True,
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
            db.add(
                MemoryEvent(
                    memory_id=old.id,
                    event_type="superseded",
                    old_value=old.memory_value,
                    new_value=memory_value,
                    reason="同一记忆键被用户明确更新",
                    source_message_id=source_message_id,
                )
            )

        memory = AgentMemory(
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            memory_key=memory_key,
            memory_value=memory_value,
            confidence=confidence,
            status="active",
            source_message_id=source_message_id,
            confirmed_at=datetime.utcnow() if confirmed else None,
        )
        db.add(memory)
        db.flush()
        db.add(
            MemoryEvent(
                memory_id=memory.id,
                event_type="created",
                old_value=None,
                new_value=memory_value,
                reason=reason,
                source_message_id=source_message_id,
            )
        )
        return memory

    def store_explicit_candidates(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: Optional[str],
        content: str,
        source_message_id: Optional[int] = None,
        use_llm: bool | None = None,
    ) -> list[MemoryAction]:
        """Extract + gate + write. Returns client-facing actions (saved/pending)."""
        payloads = collect_candidates(content, use_llm=use_llm)
        actions: list[MemoryAction] = []
        for payload in payloads:
            gated = gate_candidate(payload)
            action = self._apply_gated(
                db,
                user_id=user_id,
                session_id=session_id or "",
                source_message_id=source_message_id or 0,
                gated=gated,
            )
            if action is not None:
                actions.append(action)
        return actions

    def _apply_gated(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: str,
        source_message_id: int,
        gated: GatedMemoryCandidate,
    ) -> MemoryAction | None:
        payload = gated.payload
        if gated.decision == "reject":
            logger.info(
                "memory_rejected key=%s reason=%s",
                payload.memory_key,
                gated.reject_reason,
            )
            return None

        if gated.decision == "auto":
            memory = self.upsert(
                db,
                user_id=user_id,
                session_id=session_id or None,
                memory_type=payload.memory_type,
                memory_key=payload.memory_key,
                memory_value=payload.memory_value,
                confidence=payload.confidence,
                source_message_id=source_message_id or None,
                reason=payload.reason or "gated_auto",
                confirmed=True,
            )
            return MemoryAction(
                action="saved",
                memory_type=payload.memory_type,
                memory_key=payload.memory_key,
                memory_value=payload.memory_value,
                memory_id=memory.id,
                message=format_memory_toast("saved", payload.memory_key, payload.memory_value),
            )

        # pending — never inject into prompt until confirmed
        if not session_id or not source_message_id:
            logger.info("memory_pending_skipped missing session/message")
            return None
        candidate_row = self._create_pending_candidate(
            db,
            user_id=user_id,
            session_id=session_id,
            source_message_id=source_message_id,
            payload=payload,
        )
        return MemoryAction(
            action="pending",
            memory_type=payload.memory_type,
            memory_key=payload.memory_key,
            memory_value=payload.memory_value,
            candidate_id=candidate_row.id,
            message=format_memory_toast("pending", payload.memory_key, payload.memory_value),
        )

    def _create_pending_candidate(
        self,
        db: Session,
        *,
        user_id: str,
        session_id: str,
        source_message_id: int,
        payload: MemoryCandidatePayload,
    ) -> MemoryCandidateRow:
        # Collapse duplicate pending for same user/key.
        existing = db.scalars(
            select(MemoryCandidateRow).where(
                MemoryCandidateRow.user_id == user_id,
                MemoryCandidateRow.status == "pending",
            )
        ).all()
        for row in existing:
            try:
                data = json.loads(row.payload_json)
            except json.JSONDecodeError:
                continue
            if (
                data.get("memory_type") == payload.memory_type
                and data.get("memory_key") == payload.memory_key
            ):
                row.status = "superseded"
                row.resolved_at = datetime.utcnow()

        row = MemoryCandidateRow(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            source_message_id=source_message_id,
            payload_json=payload.model_dump_json(),
            status="pending",
            expires_at=datetime.utcnow() + timedelta(hours=MEMORY_CANDIDATE_TTL_HOURS),
        )
        db.add(row)
        db.flush()
        return row

    def list_pending_candidates(
        self,
        db: Session,
        user_id: str,
    ) -> list[dict[str, Any]]:
        now = datetime.utcnow()
        rows = db.scalars(
            select(MemoryCandidateRow)
            .where(
                MemoryCandidateRow.user_id == user_id,
                MemoryCandidateRow.status == "pending",
                MemoryCandidateRow.expires_at > now,
            )
            .order_by(MemoryCandidateRow.created_at.desc())
        ).all()
        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                data = json.loads(row.payload_json)
            except json.JSONDecodeError:
                continue
            items.append(
                {
                    "id": row.id,
                    "memory_type": data.get("memory_type"),
                    "memory_key": data.get("memory_key"),
                    "memory_value": data.get("memory_value"),
                    "confidence": data.get("confidence"),
                    "reason": data.get("reason"),
                    "status": "pending",
                    "expires_at": row.expires_at.isoformat() if row.expires_at else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            )
        return items

    def confirm_candidate(
        self,
        db: Session,
        *,
        user_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        row = db.scalar(
            select(MemoryCandidateRow).where(
                MemoryCandidateRow.id == candidate_id,
                MemoryCandidateRow.user_id == user_id,
                MemoryCandidateRow.status == "pending",
            )
        )
        if row is None:
            return {"id": candidate_id, "status": "not_found"}
        if row.expires_at and row.expires_at <= datetime.utcnow():
            row.status = "expired"
            row.resolved_at = datetime.utcnow()
            db.add(row)
            return {"id": candidate_id, "status": "expired"}

        try:
            payload = MemoryCandidatePayload.model_validate_json(row.payload_json)
        except ValidationError:
            row.status = "rejected"
            row.resolved_at = datetime.utcnow()
            db.add(row)
            return {"id": candidate_id, "status": "invalid"}

        gated = gate_candidate(payload)
        if gated.decision == "reject":
            row.status = "rejected"
            row.resolved_at = datetime.utcnow()
            db.add(row)
            return {"id": candidate_id, "status": "rejected", "reason": gated.reject_reason}

        memory = self.upsert(
            db,
            user_id=user_id,
            session_id=row.session_id,
            memory_type=payload.memory_type,
            memory_key=payload.memory_key,
            memory_value=payload.memory_value,
            confidence=payload.confidence,
            source_message_id=row.source_message_id,
            reason="用户确认待定记忆",
            confirmed=True,
        )
        row.status = "confirmed"
        row.resolved_at = datetime.utcnow()
        db.add(row)
        return {
            "id": candidate_id,
            "status": "confirmed",
            "memory_id": memory.id,
            "memory_type": payload.memory_type,
            "memory_key": payload.memory_key,
            "memory_value": payload.memory_value,
            "message": format_memory_toast("saved", payload.memory_key, payload.memory_value),
        }

    def reject_candidate(
        self,
        db: Session,
        *,
        user_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        row = db.scalar(
            select(MemoryCandidateRow).where(
                MemoryCandidateRow.id == candidate_id,
                MemoryCandidateRow.user_id == user_id,
                MemoryCandidateRow.status == "pending",
            )
        )
        if row is None:
            return {"id": candidate_id, "status": "not_found"}
        row.status = "rejected"
        row.resolved_at = datetime.utcnow()
        db.add(row)
        return {"id": candidate_id, "status": "rejected"}

    def render_active_context(
        self,
        db: Session,
        user_id: str,
        *,
        query: str | None = None,
        limit: int = 5,
    ) -> str:
        """Render a small profile block for the system prompt.

        Only active AgentMemory rows are injected — pending candidates never appear.
        """
        cap = max(0, int(limit))
        if cap == 0:
            return ""

        memories = list(
            db.scalars(
                select(AgentMemory)
                .where(AgentMemory.user_id == user_id, AgentMemory.status == "active")
                .order_by(AgentMemory.updated_at.desc())
                .limit(40)
            ).all()
        )
        if not memories:
            return ""

        q = (query or "").strip().lower()

        def _score(memory: AgentMemory) -> tuple[int, float]:
            key = (memory.memory_key or "").lower()
            value = (memory.memory_value or "").lower()
            aliases = _MEMORY_KEY_ALIASES.get(key, ())
            blob = f"{key} {value} {' '.join(aliases)}"
            overlap = 0
            if q:
                if key and key in q:
                    overlap += 3
                if value and value in q:
                    overlap += 2
                for alias in aliases:
                    if alias and alias in q:
                        overlap += 3
                for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", q):
                    if token in blob:
                        overlap += 1
            return (overlap, float(memory.confidence or 0.0))

        ranked = sorted(memories, key=_score, reverse=True)
        chosen = ranked[:cap]
        return "\n".join(
            f"[{memory.memory_type}] {memory.memory_key}: {memory.memory_value}"
            for memory in chosen
        )

    def audit_event_count(self, db: Session, memory_id: int) -> int:
        return int(
            db.scalar(
                select(func.count())
                .select_from(MemoryEvent)
                .where(MemoryEvent.memory_id == memory_id)
            )
            or 0
        )
