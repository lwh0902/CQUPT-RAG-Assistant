from collections.abc import Iterator
import os

# Ensure secret validation passes for unit tests before app modules import.
os.environ.setdefault("JWT_SECRET_KEY", "unit-test-jwt-secret-key-32chars-min")
os.environ.setdefault("MYSQL_USER", "test")
os.environ.setdefault("MYSQL_PASSWORD", "unit-test-strong-password")
os.environ.setdefault("ALLOW_INSECURE_DB_PASSWORD", "false")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "7")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database import Base
import models  # noqa: F401 - Registers ORM models before metadata creation.


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
