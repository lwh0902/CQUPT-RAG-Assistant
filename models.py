from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class User(Base):
    """用户表：系统中的用户主体。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    phone: Mapped[str] = mapped_column(String(20), unique=True, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=True)

    sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    model_settings: Mapped[Optional["UserModelSettings"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
    )


class UserModelSettings(Base):
    """每位用户独立保存的回答模型参数。"""

    __tablename__ = "user_model_settings"

    user_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    temperature: Mapped[float] = mapped_column(nullable=False, default=0.3)
    top_p: Mapped[float] = mapped_column(nullable=False, default=0.8)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user: Mapped["User"] = relationship(back_populates="model_settings")


class AgentMemory(Base):
    """可覆盖的稳定用户记忆，不承担知识库事实来源角色。"""

    __tablename__ = "agent_memories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)
    memory_key: Mapped[str] = mapped_column(String(100), nullable=False)
    memory_value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    scope: Mapped[str] = mapped_column(String(20), nullable=False, default="global")
    source_message_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class MemoryEvent(Base):
    """稳定记忆的创建、覆盖和删除审计记录。"""

    __tablename__ = "memory_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("agent_memories.id", ondelete="CASCADE"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    source_message_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class MemoryCandidate(Base):
    """A model-proposed memory that cannot affect context until the user confirms it."""

    __tablename__ = "memory_candidates"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(50), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(50), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    source_message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class InterviewSession(Base):
    """面试助手：一次 JD + 公司 + 简历的题库生成记录。"""

    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    company: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    position: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    jd_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    resume_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    resume_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    reference_used: Mapped[bool] = mapped_column(nullable=False, default=False)
    reference_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    report_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    questions: Mapped[List["InterviewQuestion"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class InterviewQuestion(Base):
    """面试助手：单条题目（选择题 mcq / 简答题 qa），内容为 JSON。"""

    __tablename__ = "interview_questions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    qtype: Mapped[str] = mapped_column(String(10), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    round: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    session: Mapped["InterviewSession"] = relationship(back_populates="questions")


class UserUsageDaily(Base):
    """Per-user daily counters used by the single-instance quota service."""

    __tablename__ = "user_usage_daily"
    __table_args__ = (UniqueConstraint("user_id", "usage_date", name="uq_user_usage_daily"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    usage_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    chat_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    web_search_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ModelCallAudit(Base):
    """Operational record for an LLM invocation without storing prompts or secrets."""

    __tablename__ = "model_call_audits"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    call_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    web_search_enabled: Mapped[str] = mapped_column(String(5), nullable=False, default="false")
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provider_request_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_summary: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


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

    # Working-memory overflow: one rolling summary per session (MySQL, not vector DB).
    overflow_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    overflow_until_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    overflow_from_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    overflow_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

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
