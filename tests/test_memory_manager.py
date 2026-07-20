from models import User
from services.memory_manager import MemoryManager


def test_explicit_memory_update_supersedes_old_value_and_records_audit(db_session) -> None:
    user = User(id="user-1", username="user_1")
    db_session.add(user)
    db_session.commit()
    manager = MemoryManager()

    first = manager.upsert(
        db_session,
        user_id=user.id,
        session_id="session-1",
        memory_type="preference",
        memory_key="answer_style",
        memory_value="简洁",
        confidence=0.95,
    )
    second = manager.upsert(
        db_session,
        user_id=user.id,
        session_id="session-1",
        memory_type="preference",
        memory_key="answer_style",
        memory_value="详细",
        confidence=0.95,
    )
    db_session.commit()

    assert first.status == "superseded"
    assert second.status == "active"
    assert manager.audit_event_count(db_session, second.id) == 1


def test_ambiguous_message_does_not_create_stable_memory(db_session) -> None:
    manager = MemoryManager()

    candidates = manager.extract_explicit_candidates("今天心情一般。")

    assert candidates == []
