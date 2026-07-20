"""Application logging for local development and production process managers."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from services.log_context import ContextFilter, SensitiveDataFilter


_FORMAT = "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s user_id=%(user_id)s session_id=%(session_id)s %(message)s"


def configure_logging(*, log_dir: Path, level: str = "INFO", backup_count: int = 14) -> logging.Logger:
    """Configure console and daily rolling file handlers without duplicating them."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(root.handlers):
        if getattr(handler, "_cqupt_rag_managed", False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(ContextFilter())
    console.addFilter(SensitiveDataFilter())
    console._cqupt_rag_managed = True

    file_handler = TimedRotatingFileHandler(
        log_dir / "cqupt-rag.log",
        when="midnight",
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ContextFilter())
    file_handler.addFilter(SensitiveDataFilter())
    file_handler._cqupt_rag_managed = True
    root.addHandler(console)
    root.addHandler(file_handler)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    return logging.getLogger("cqupt_rag")
