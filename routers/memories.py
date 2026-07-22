"""Account-scoped memory management with auditable soft deletion and pending confirm."""

from __future__ import annotations

from typing import Any, Union

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from models import AgentMemory, MemoryEvent, User
from security import get_current_user
from services.memory_manager import MemoryManager

router = APIRouter(prefix="/memories", tags=["memories"])
memory_manager = MemoryManager()


def _serialize(memory: AgentMemory) -> dict[str, object]:
    return {
        "id": memory.id,
        "memory_type": memory.memory_type,
        "memory_key": memory.memory_key,
        "memory_value": memory.memory_value,
        "confidence": memory.confidence,
        "status": memory.status,
    }


@router.get("")
def list_memories(
    current_user: User = Depends(get_current_user),
) -> dict[str, list[dict[str, object]]]:
    with Session(engine) as db:
        memories = db.scalars(
            select(AgentMemory)
            .where(AgentMemory.user_id == current_user.id, AgentMemory.status == "active")
            .order_by(AgentMemory.updated_at.desc(), AgentMemory.id.desc())
        ).all()
        pending = memory_manager.list_pending_candidates(db, current_user.id)
        return {
            "memories": [_serialize(memory) for memory in memories],
            "pending": pending,
        }


@router.get("/pending")
def list_pending_memories(
    current_user: User = Depends(get_current_user),
) -> dict[str, list[dict[str, Any]]]:
    with Session(engine) as db:
        return {"pending": memory_manager.list_pending_candidates(db, current_user.id)}


@router.post("/candidates/{candidate_id}/confirm")
def confirm_memory_candidate(
    candidate_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    with Session(engine) as db:
        result = memory_manager.confirm_candidate(
            db,
            user_id=current_user.id,
            candidate_id=candidate_id,
        )
        db.commit()
        return result


@router.post("/candidates/{candidate_id}/reject")
def reject_memory_candidate(
    candidate_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    with Session(engine) as db:
        result = memory_manager.reject_candidate(
            db,
            user_id=current_user.id,
            candidate_id=candidate_id,
        )
        db.commit()
        return result


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: int,
    current_user: User = Depends(get_current_user),
) -> dict[str, Union[int, str]]:
    with Session(engine) as db:
        memory = db.scalar(
            select(AgentMemory).where(
                AgentMemory.id == memory_id,
                AgentMemory.user_id == current_user.id,
                AgentMemory.status == "active",
            )
        )
        if memory is None:
            return {"id": memory_id, "status": "not_found"}

        previous_value = memory.memory_value
        memory.status = "deleted"
        db.add(
            MemoryEvent(
                memory_id=memory.id,
                event_type="deleted",
                old_value=previous_value,
                new_value=None,
                reason="用户主动删除记忆",
                source_message_id=None,
            )
        )
        db.commit()
        return {"id": memory_id, "status": "deleted"}
