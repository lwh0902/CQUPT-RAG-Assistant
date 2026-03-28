from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from models import User


def main() -> None:
    with Session(engine) as db:
        stmt = select(User).where(User.username == "重邮极客")
        user = db.scalar(stmt)

        if user is None:
            print("未找到用户：重邮极客")
            return

        print(f"用户：{user.username}")
        for session in user.sessions:
            print(f"\n会话标题：{session.title}")
            for message in session.messages:
                print(f"[{message.role}]: {message.content}")


if __name__ == "__main__":
    main()
