from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker


MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "cqupt_rag")

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=int(MYSQL_PORT),
    database=MYSQL_DATABASE,
    query={"charset": "utf8mb4"},
)

engine = create_engine(
    DATABASE_URL,
    echo=True,
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
        echo=True,
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
