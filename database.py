from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker


load_dotenv()

_WEAK_MYSQL_PASSWORDS = {
    "",
    "123456",
    "password",
    "root",
    "mysql",
    "admin",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def assert_safe_database_password(password: str | None, *, allow_insecure: bool) -> None:
    """Refuse known-weak MySQL passwords unless explicitly allowed for local dev."""
    value = (password or "").strip()
    if allow_insecure:
        return
    if value.lower() in _WEAK_MYSQL_PASSWORDS or len(value) < 8:
        raise RuntimeError(
            "MYSQL_PASSWORD is missing or too weak for deployment. "
            "Use a strong password with at least 8 characters, "
            "or set ALLOW_INSECURE_DB_PASSWORD=true only for local development."
        )


MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "cqupt_rag")
SQL_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"
ALLOW_INSECURE_DB_PASSWORD = _env_flag("ALLOW_INSECURE_DB_PASSWORD", default=False)

if not MYSQL_USER:
    raise RuntimeError("MYSQL_USER is not set. Add it to your environment or .env file.")

assert_safe_database_password(
    MYSQL_PASSWORD,
    allow_insecure=ALLOW_INSECURE_DB_PASSWORD,
)

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=MYSQL_USER,
    password=MYSQL_PASSWORD or "",
    host=MYSQL_HOST,
    port=int(MYSQL_PORT),
    database=MYSQL_DATABASE,
    query={"charset": "utf8mb4"},
)

engine = create_engine(
    DATABASE_URL,
    echo=SQL_ECHO,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def ensure_database_exists() -> None:
    server_engine = create_engine(
        DATABASE_URL.set(database=None),
        echo=SQL_ECHO,
        pool_pre_ping=True,
        isolation_level="AUTOCOMMIT",
    )
    escaped_database_name = MYSQL_DATABASE.replace("`", "``")

    with server_engine.connect() as connection:
        connection.execute(
            text(
                "CREATE DATABASE IF NOT EXISTS "
                f"`{escaped_database_name}` CHARACTER SET utf8mb4"
            )
        )

    server_engine.dispose()
