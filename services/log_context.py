"""Per-request logging context and sensitive value redaction."""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar, Token
from typing import Dict, Optional


_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_user_id: ContextVar[str] = ContextVar("user_id", default="-")
_session_id: ContextVar[str] = ContextVar("session_id", default="-")

_PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(password|token|api[_-]?key|secret|authorization|resume[_-]?text|jd[_-]?text|invite[_-]?code)\b\s*([=:])\s*([^\s,;]+)"
)
_LONG_PII_PATTERN = re.compile(
    r"(?i)\b(resume_text|jd_text|spoken_answer)\b\s*([=:])\s*['\"]?.{40,}"
)


def _masked_identifier(value: Optional[str]) -> str:
    if not value:
        return "-"
    if len(value) <= 4:
        return "***"
    return "{}...{}".format(value[:4], value[-4:])


def bind_log_context(
    *,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Token]:
    """Bind values for the current request/task and return reset tokens."""
    tokens: Dict[str, Token] = {}
    if request_id is not None:
        tokens["request_id"] = _request_id.set(request_id)
    if user_id is not None:
        tokens["user_id"] = _user_id.set(_masked_identifier(user_id))
    if session_id is not None:
        tokens["session_id"] = _session_id.set(_masked_identifier(session_id))
    return tokens


def reset_log_context(tokens: Dict[str, Token]) -> None:
    variables = {
        "request_id": _request_id,
        "user_id": _user_id,
        "session_id": _session_id,
    }
    for name, token in tokens.items():
        variables[name].reset(token)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.user_id = _user_id.get()
        record.session_id = _session_id.get()
        return True


class SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        message = _PHONE_PATTERN.sub("1**********", message)
        message = _BEARER_PATTERN.sub(r"\1[REDACTED]", message)
        message = _SECRET_PATTERN.sub(r"\1\2[REDACTED]", message)
        message = _LONG_PII_PATTERN.sub(r"\1\2[REDACTED_PII]", message)
        record.msg = message
        record.args = ()
        return True
