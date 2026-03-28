from __future__ import annotations

import uuid
from typing import List

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class User(Base):
    """用户表：系统中的用户主体。"""

    __tablename__ = "users"

    # 对外暴露的主键改为 UUID 字符串，降低被枚举和越权猜测的风险。
    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # 一个用户可以拥有多个聊天会话，删除用户时级联删除其所有会话。
    sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class ChatSession(Base):
    """聊天会话表：承载某个用户的一组连续对话。"""

    __tablename__ = "sessions"

    # 会话主键同样改为 UUID 字符串，适合暴露给前端作为会话标识。
    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    # 外键类型必须和 users.id 保持一致，才能确保物理关联正确。
    user_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 多对一：当前会话归属于一个用户。
    user: Mapped["User"] = relationship(back_populates="sessions")

    # 一对多：一个会话下包含多条消息，删除会话时级联删除消息。
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """消息表：保存用户与 AI 的每一条对话消息。"""

    __tablename__ = "messages"

    # 消息主键保留整数自增，适合内部存储和排序，不直接暴露给前端。
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 外键类型和 sessions.id 对齐，改为 UUID 字符串。
    session_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 多对一：当前消息归属于一个聊天会话。
    session: Mapped["ChatSession"] = relationship(back_populates="messages")
