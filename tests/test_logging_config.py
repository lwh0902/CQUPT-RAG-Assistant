import logging

import database
from services.log_context import bind_log_context, reset_log_context
from services.logging_config import configure_logging


def test_configure_logging_writes_application_events_to_daily_file(tmp_path) -> None:
    logger = configure_logging(log_dir=tmp_path, level="INFO")

    logger.info("memory deletion completed")
    for handler in logging.getLogger().handlers:
        handler.flush()

    log_files = list(tmp_path.glob("cqupt-rag.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8")
    assert "INFO" in content
    assert "memory deletion completed" in content


def test_configure_logging_does_not_emit_sql_parameters(tmp_path) -> None:
    configure_logging(log_dir=tmp_path, level="INFO")

    assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING


def test_sql_echo_is_disabled_by_default() -> None:
    assert database.SQL_ECHO is False


def test_contextual_logs_include_correlation_fields_and_redact_sensitive_values(tmp_path) -> None:
    logger = configure_logging(log_dir=tmp_path, level="INFO")
    tokens = bind_log_context(request_id="req-123", user_id="user-456", session_id="session-789")
    try:
        logger.warning(
            "phone=18128161378 authorization=Bearer top-secret password=hunter2 api_key=abc123"
        )
    finally:
        reset_log_context(tokens)
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = (tmp_path / "cqupt-rag.log").read_text(encoding="utf-8")
    assert "request_id=req-123" in content
    assert "user_id=user...-456" in content
    assert "session_id=sess...-789" in content
    assert "user_id=user-456" not in content
    assert "session_id=session-789" not in content
    assert "18128161378" not in content
    assert "top-secret" not in content
    assert "hunter2" not in content
    assert "abc123" not in content
