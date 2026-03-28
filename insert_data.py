from sqlalchemy.orm import Session

from database import engine
from models import ChatSession, Message, User


def main() -> None:
    print("开始写入 RAG AI 记忆示例数据...")

    with Session(engine) as db, db.begin():
        student = User(username="重邮极客")
        lesson_session = ChatSession(title="RAG 架构第一课", user=student)
        Message(
            role="user",
            content="什么是 RAG，为什么它适合做企业知识库问答？",
            session=lesson_session,
        )
        Message(
            role="ai",
            content="RAG 会先检索外部知识，再把结果注入大模型生成过程，因此特别适合知识库问答。",
            session=lesson_session,
        )

        db.add(student)

    print("示例数据写入成功，用户、会话和消息已级联保存到 MySQL。")


if __name__ == "__main__":
    main()
