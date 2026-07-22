from database import Base, engine, ensure_database_exists, ensure_session_overflow_columns
import models  # noqa: F401


def main() -> None:
    ensure_database_exists()
    Base.metadata.create_all(bind=engine)
    ensure_session_overflow_columns()
    print("RAG AI memory database schema created successfully.")


if __name__ == "__main__":
    main()
