from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import routers.memories as memories_router
from database import Base
from models import AgentMemory, MemoryEvent, User


def _memory(user_id: str, value: str) -> AgentMemory:
    return AgentMemory(
        user_id=user_id,
        memory_type="preference",
        memory_key="answer_style",
        memory_value=value,
        confidence=0.95,
        status="active",
    )


def test_list_memories_returns_only_current_users_active_memories(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(memories_router, "engine", engine)

    with Session(engine) as db:
        db.add_all([User(id="user-1", username="one"), User(id="user-2", username="two")])
        db.add_all([_memory("user-1", "简洁"), _memory("user-2", "详细")])
        db.commit()

    result = memories_router.list_memories(current_user=User(id="user-1", username="one"))

    assert result["memories"] == [
        {
            "id": 1,
            "memory_type": "preference",
            "memory_key": "answer_style",
            "memory_value": "简洁",
            "confidence": 0.95,
        }
    ]


def test_delete_memory_marks_only_owners_memory_inactive_and_preserves_audit(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(memories_router, "engine", engine)

    with Session(engine) as db:
        db.add_all([User(id="user-1", username="one"), User(id="user-2", username="two")])
        owned = _memory("user-1", "简洁")
        other = _memory("user-2", "详细")
        db.add_all([owned, other])
        db.flush()
        db.add(MemoryEvent(
            memory_id=owned.id,
            event_type="created",
            old_value=None,
            new_value="简洁",
            reason="用户明确表达的稳定信息",
        ))
        db.commit()
        owned_id, other_id = owned.id, other.id

    result = memories_router.delete_memory(owned_id, current_user=User(id="user-1", username="one"))

    assert result == {"id": owned_id, "status": "deleted"}
    with Session(engine) as db:
        owned = db.get(AgentMemory, owned_id)
        other = db.get(AgentMemory, other_id)
        events = db.scalars(
            select(MemoryEvent).where(MemoryEvent.memory_id == owned_id).order_by(MemoryEvent.id)
        ).all()
        assert owned.status == "deleted"
        assert other.status == "active"
        assert [event.event_type for event in events] == ["created", "deleted"]


def test_delete_memory_does_not_expose_another_users_memory(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(memories_router, "engine", engine)

    with Session(engine) as db:
        db.add_all([User(id="user-1", username="one"), User(id="user-2", username="two")])
        other = _memory("user-2", "详细")
        db.add(other)
        db.commit()
        other_id = other.id

    result = memories_router.delete_memory(other_id, current_user=User(id="user-1", username="one"))

    assert result == {"id": other_id, "status": "not_found"}
    with Session(engine) as db:
        assert db.get(AgentMemory, other_id).status == "active"
