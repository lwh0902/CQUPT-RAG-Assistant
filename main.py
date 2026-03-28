from database import Base, engine, ensure_database_exists
import models  # noqa: F401


def main() -> None:
    ensure_database_exists()
    Base.metadata.create_all(bind=engine)
    print("RAG AI memory database schema created successfully.")


if __name__ == "__main__":
    main()
