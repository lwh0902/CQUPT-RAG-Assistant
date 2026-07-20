import models
from datetime import datetime, timedelta


def test_model_settings_are_account_scoped_with_defaults(db_session) -> None:
    assert hasattr(models, "UserModelSettings")

    user = models.User(id="user-1", username="user_1")
    settings = models.UserModelSettings(user_id=user.id)
    db_session.add_all([user, settings])
    db_session.commit()
    db_session.refresh(settings)

    assert settings.user_id == user.id
    assert settings.temperature == 0.3
    assert settings.top_p == 0.8


def test_audit_usage_and_memory_candidate_models_have_required_lifecycle_fields(db_session) -> None:
    assert hasattr(models, "ModelCallAudit")
    assert hasattr(models, "UserUsageDaily")
    assert hasattr(models, "MemoryCandidate")

    user = models.User(id="user-2", username="user_2")
    session = models.ChatSession(id="session-22222", title="测试", user_id=user.id)
    message = models.Message(role="user", content="以后回答简洁", session_id=session.id)
    db_session.add_all([user, session, message])
    db_session.flush()

    candidate = models.MemoryCandidate(
        id="candidate-1",
        user_id=user.id,
        session_id=session.id,
        source_message_id=message.id,
        payload_json='{"memory_value":"简洁"}',
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    db_session.add(candidate)
    db_session.commit()

    assert candidate.status == "pending"
    assert candidate.expires_at > datetime.utcnow()
