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


def test_render_active_context_respects_limit_and_query_relevance(db_session) -> None:
    user = User(id="user-2", username="user_2")
    db_session.add(user)
    db_session.commit()
    manager = MemoryManager()

    manager.upsert(
        db_session,
        user_id=user.id,
        session_id="s1",
        memory_type="profile",
        memory_key="major",
        memory_value="计算机",
        confidence=0.9,
    )
    manager.upsert(
        db_session,
        user_id=user.id,
        session_id="s1",
        memory_type="preference",
        memory_key="answer_style",
        memory_value="简洁",
        confidence=0.9,
    )
    manager.upsert(
        db_session,
        user_id=user.id,
        session_id="s1",
        memory_type="profile",
        memory_key="college",
        memory_value="光电工程学院",
        confidence=0.9,
    )
    db_session.commit()

    limited = manager.render_active_context(db_session, user.id, limit=2)
    assert limited.count("\n") + (1 if limited else 0) <= 2

    focused = manager.render_active_context(
        db_session,
        user.id,
        query="我的专业课怎么选",
        limit=1,
    )
    assert "major" in focused or "计算机" in focused

    empty = manager.render_active_context(db_session, user.id, limit=0)
    assert empty == ""


def test_store_explicit_auto_saves_high_confidence(db_session) -> None:
    from models import ChatSession, Message

    user = User(id="user-3", username="user_3")
    db_session.add(user)
    db_session.add(ChatSession(id="session-3", title="t", user_id=user.id))
    msg = Message(role="user", content="我是软件工程专业", session_id="session-3")
    db_session.add(msg)
    db_session.commit()

    manager = MemoryManager()
    actions = manager.store_explicit_candidates(
        db_session,
        user_id=user.id,
        session_id="session-3",
        content="我是软件工程专业，以后请简洁回答",
        source_message_id=msg.id,
        use_llm=False,
    )
    db_session.commit()

    assert any(a.action == "saved" for a in actions)
    rendered = manager.render_active_context(db_session, user.id, limit=5)
    assert "软件工程" in rendered or "简洁" in rendered


def test_pending_candidate_not_in_prompt_until_confirmed(db_session) -> None:
    from models import ChatSession, Message
    from services.memory_candidates import MemoryCandidatePayload
    from services.memory_candidates import gate_candidate

    user = User(id="user-4", username="user_4")
    db_session.add(user)
    db_session.add(ChatSession(id="session-4", title="t", user_id=user.id))
    msg = Message(role="user", content="x", session_id="session-4")
    db_session.add(msg)
    db_session.commit()

    manager = MemoryManager()
    payload = MemoryCandidatePayload(
        memory_type="profile",
        memory_key="major",
        memory_value="光电信息",
        confidence=0.8,
        explicit=True,
    )
    assert gate_candidate(payload).decision == "pending"

    # Force pending path via internal apply
    from services.memory_candidates import GatedMemoryCandidate

    action = manager._apply_gated(
        db_session,
        user_id=user.id,
        session_id="session-4",
        source_message_id=msg.id,
        gated=GatedMemoryCandidate(payload, "pending"),
    )
    db_session.commit()
    assert action is not None and action.action == "pending" and action.candidate_id

    assert manager.render_active_context(db_session, user.id) == ""

    confirmed = manager.confirm_candidate(
        db_session, user_id=user.id, candidate_id=action.candidate_id
    )
    db_session.commit()
    assert confirmed["status"] == "confirmed"
    assert "光电信息" in manager.render_active_context(db_session, user.id)
