"""Chat routes: REST endpoint and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session

from database import engine
from models import ChatSession, Message, User, UserModelSettings
from settings import (
    CHROMA_SESSION_SUMMARY_IN_PROMPT,
    MODEL_NAME,
    PROFILE_INJECT_K,
    SESSION_HISTORY_LOAD_LIMIT,
    SHORT_TERM_MESSAGE_LIMIT,
    SHORT_TERM_ROUNDS,
)
from security import get_current_user
from services.confidence import calculate_confidence
from services.conversation_context import (
    previous_user_utterance,
    resolve_followup_query,
    resolve_followup_query_llm,
    turns_to_chat_messages,
)
from services.llm import create_llm_completion, get_llm_client, stream_llm_text
from services.memory_manager import MemoryManager
from services.retrieval import collect_evidence, format_web_evidence_for_prompt
from services.log_context import bind_log_context, reset_log_context
from services.working_memory import load_or_refresh_overflow_summary, split_near_overflow
from tools import AVAILABLE_TOOLS_MAP

router = APIRouter(tags=["chat"])

MESSAGES_PER_ROUND = 2
LONG_TERM_MEMORY_DIR = Path("./chroma_memory_db")
LONG_TERM_MEMORY_COLLECTION = "chat_long_term_memory"
DEFAULT_SESSION_TITLE = "新建对话"
SESSION_TITLE_LIMIT = 18
memory_manager = MemoryManager()


class ChatRequest(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        min_length=5,
        max_length=50,
        description="当前请求用户的唯一标识",
    )
    session_id: str = Field(
        ...,
        min_length=10,
        max_length=50,
        description="聊天窗口的唯一标识",
    )
    new_message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户最新发送的聊天内容",
    )
    web_search_enabled: bool = Field(
        default=False,
        description="是否允许本次消息调用联网搜索工具",
    )

    @field_validator("new_message")
    @classmethod
    def validate_new_message(cls, value: str) -> str:
        blocked_keywords = (
            "system prompt",
            "ignore all previous",
            "forget everything",
        )
        lowered = value.lower()
        if any(keyword in lowered for keyword in blocked_keywords):
            raise ValueError("检测到非法指令，已拦截可疑 Prompt 注入。")
        return value


def _build_session_title(message: str) -> str:
    compact_message = " ".join((message or "").strip().split())
    if not compact_message:
        return DEFAULT_SESSION_TITLE
    if len(compact_message) <= SESSION_TITLE_LIMIT:
        return compact_message
    return f"{compact_message[:SESSION_TITLE_LIMIT]}..."


def ensure_session_exists(request: ChatRequest) -> None:
    if not request.user_id:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")

    with Session(engine) as db:
        current_user = db.scalar(select(User).where(User.id == request.user_id))
        if current_user is None:
            raise HTTPException(status_code=401, detail="未登录或登录已过期")

        chat_window = db.scalar(
            select(ChatSession).where(ChatSession.id == request.session_id)
        )

        if chat_window is None:
            chat_window = ChatSession(
                id=request.session_id,
                title=DEFAULT_SESSION_TITLE,
                user_id=current_user.id,
            )
            db.add(chat_window)
            db.commit()
            db.refresh(chat_window)
            return

        if chat_window.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问该聊天会话。")


def save_messages(request: ChatRequest, ai_reply: str) -> int:
    with Session(engine) as db:
        chat_window = db.scalar(
            select(ChatSession).where(ChatSession.id == request.session_id)
        )
        if (
            chat_window is not None
            and (not chat_window.title.strip() or chat_window.title == DEFAULT_SESSION_TITLE)
        ):
            chat_window.title = _build_session_title(request.new_message)

        user_message = Message(
            role="user",
            content=request.new_message,
            session_id=request.session_id,
        )
        ai_message = Message(
            role="assistant",
            content=ai_reply,
            session_id=request.session_id,
        )
        db.add_all([user_message, ai_message])
        db.commit()
        return user_message.id


def persist_explicit_memories(
    user_id: str,
    session_id: str,
    content: str,
    source_message_id: int,
) -> list[dict[str, Any]]:
    """Extract/gate memories; return client-facing actions for toast/confirm UI."""
    with Session(engine) as db:
        actions = memory_manager.store_explicit_candidates(
            db,
            user_id=user_id,
            session_id=session_id,
            content=content,
            source_message_id=source_message_id,
        )
        db.commit()
        return [action.to_dict() for action in actions]


def get_generation_options(user_id: str) -> dict[str, float]:
    with Session(engine) as db:
        settings = db.scalar(
            select(UserModelSettings).where(UserModelSettings.user_id == user_id)
        )
        if settings is None:
            return {"temperature": 0.3, "top_p": 0.8}
        return {"temperature": settings.temperature, "top_p": settings.top_p}


def build_system_prompt(hybrid_context: dict[str, Any]) -> str:
    knowledge = hybrid_context.get("knowledge") or "无"
    web = hybrid_context.get("web") or "无"
    long_term = hybrid_context.get("long_term") or ""
    overflow_summary = (hybrid_context.get("overflow_summary") or "").strip()
    resolved_query = hybrid_context.get("resolved_query") or ""
    resolved_block = (
        f"【当前问题（含追问消解）】\n{resolved_query}\n\n" if resolved_query else ""
    )
    overflow_block = (
        f"【会话早前要点】\n{overflow_summary}\n\n" if overflow_summary else ""
    )
    long_term_block = f"【长期记忆】\n{long_term}\n\n" if long_term else ""

    return (
        '你是“重邮极客 Agent”，是一个校园综合助手。\n'
        "你的能力包括：\n"
        "1. 基于学生手册知识回答校内制度、部门、办事流程、学籍、奖助、纪律等问题；\n"
        "2. 在需要时调用工具查询天气；\n"
        "3. 在需要时调用工具查询课表。\n"
        '当用户打招呼、寒暄，或者询问“你能做什么”时，'
        "请明确告诉用户你既可以查天气和课表，也可以检索并解答学生手册相关内容。\n"
        "请结合对话历史理解追问、指代和省略（例如“那不回呢”应承接上一问的主题）。\n"
        "若提供了会话早前要点，它仅概括更早轮次；最近几轮原文在对话消息中；"
        "不要把早前要点当作学生手册条文。\n"
        "请严格基于以下参考资料回答问题；如果问题需要实时天气或课表信息，就结合工具结果回答。\n"
        "网络资料只能引用其中真实提供的链接，不得把自身常识伪装成联网检索结果。"
        "资料不足或来源冲突时，必须明确说明不确定性。\n"
        "不要主动推荐课表或天气，除非用户明确在问这些。\n\n"
        f"{resolved_block}"
        f"{overflow_block}"
        f"{long_term_block}"
        f"【学生手册知识】\n{knowledge}\n\n"
        f"【联网检索资料】\n{web}"
    )


def build_final_messages(
    request: ChatRequest,
    hybrid_context: dict[str, Any],
) -> list[dict[str, Any]]:
    history = turns_to_chat_messages(
        hybrid_context.get("recent_turns") or [],
        exclude_last_user=True,
    )
    # Keep prompt bounded: last few turns only.
    history = history[-SHORT_TERM_MESSAGE_LIMIT:]
    return [
        {"role": "system", "content": build_system_prompt(hybrid_context)},
        *history,
        {"role": "user", "content": request.new_message},
    ]


# ---------------------------------------------------------------------------
# Intent router — skip RAG for casual chat
# ---------------------------------------------------------------------------

CHAT_PATTERNS = (
    "你好", "嗨", "hi", "hello", "hey", "早上好", "下午好", "晚上好",
    "再见", "拜拜", "谢谢", "感谢", "好的", "嗯", "哈哈", "哦",
)

CASUAL_KEYWORDS = (
    "你是谁", "你叫什么", "你能做什么", "你会什么", "你有什么功能",
    "你能帮我", "介绍一下你自己", "自我介绍",
)


def _quick_intent_check(query: str) -> Optional[str]:
    """Fast keyword-based check. Returns 'chat' or None (needs further check)."""
    stripped = query.strip()
    if not stripped:
        return None

    if stripped.lower() in CHAT_PATTERNS:
        return "chat"

    if len(stripped) <= 4 and any(kw in stripped for kw in ("你好", "嗨", "hi", "嗯", "哦")):
        return "chat"

    if any(kw in stripped for kw in CASUAL_KEYWORDS):
        return "chat"

    return None


async def classify_intent(query: str, *, previous_user: str | None = None) -> str:
    """Classify user intent as 'chat' (casual) or 'knowledge' (needs RAG).

    Uses a fast keyword pre-check first, falls back to a lightweight LLM call.
    Follow-up fragments are classified with the previous user turn in mind.
    """
    from services.conversation_context import is_followup_utterance, resolve_followup_query

    effective = resolve_followup_query(query, previous_user) if previous_user else query
    # Short deictic follow-ups after a knowledge question stay on the knowledge path.
    if previous_user and is_followup_utterance(query, previous_user):
        prev_intent_hint = _quick_intent_check(previous_user)
        if prev_intent_hint != "chat":
            return "knowledge"

    quick = _quick_intent_check(effective)
    if quick is not None:
        return quick

    prompt = (
        "判断以下用户消息的意图。只回答一个词：\n"
        "- 如果是打招呼、寒暄、闲聊、问你是谁、问你能做什么、表达感谢等日常对话 → 回答 chat\n"
        "- 如果是关于学生手册、校规、学籍、奖学金、处分、考试、请假、课表、天气等具体问题 → 回答 knowledge\n"
        "- 如果是对上一问的追问/省略（如“那不回呢”“如果更严重呢”）且上一问是制度问题 → 回答 knowledge\n\n"
        + (f"上一轮用户问题：{previous_user}\n" if previous_user else "")
        + f"用户消息：{query}\n\n意图："
    )

    def _call() -> str:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return (response.choices[0].message.content or "").strip().lower()

    result = await asyncio.to_thread(_call)
    return result if result in ("chat", "knowledge") else "knowledge"


CHAT_SYSTEM_PROMPT = (
    '你是"重邮极客 Agent"，是一个校园综合助手。\n'
    "你的能力包括：\n"
    "1. 基于学生手册知识回答校内制度、部门、办事流程、学籍、奖助、纪律等问题；\n"
    "2. 在需要时调用工具查询天气；\n"
    "3. 在需要时调用工具查询课表。\n"
    "请友好、简洁地回复用户。如果用户想查询具体制度或信息，提示他们可以直接提问。"
)


def load_recent_turns(
    session_id: str,
    *,
    limit: int = SESSION_HISTORY_LOAD_LIMIT,
) -> list[dict[str, Any]]:
    """Load chat turns oldest→newest (includes message id for overflow cache keys)."""
    with Session(engine) as db:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.id))
            .limit(limit)
        )
        rows = list(reversed(db.scalars(stmt).all()))
    turns: list[dict[str, Any]] = []
    for row in rows:
        role = getattr(row, "role", None)
        content = (getattr(row, "content", None) or "").strip()
        if role in {"user", "assistant"} and content:
            turns.append(
                {
                    "id": getattr(row, "id", None),
                    "role": role,
                    "content": content,
                }
            )
    return turns


def build_chat_messages(
    query: str,
    *,
    recent_turns: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build messages for casual chat (no RAG context needed)."""
    history = turns_to_chat_messages(recent_turns or [], exclude_last_user=True)
    history = history[-SHORT_TERM_MESSAGE_LIMIT:]
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": query},
    ]


def build_sources_from_knowledge(knowledge_text: str) -> list[dict[str, Any]]:
    """Extract page-level source snippets from formatted RAG context."""
    if not knowledge_text.strip():
        return []

    from settings import POLICY_DOCUMENTS

    document_name_to_id = {
        item["document_name"]: item["document_id"] for item in POLICY_DOCUMENTS
    }

    source_map: dict[tuple[str, int], str] = {}
    pattern = re.compile(
        r"【(.+?)｜第\s*(\d+)\s*页】\s*\n?(.*?)(?=\n\n【.+?｜第\s*\d+\s*页】|\Z)",
        re.S,
    )

    for match in pattern.finditer(knowledge_text):
        document_name = match.group(1).strip()
        page = int(match.group(2))
        snippet = " ".join(match.group(3).strip().split())
        if not snippet:
            continue
        source_map.setdefault((document_name, page), snippet[:420])

    return [
        {
            "document_id": document_name_to_id.get(document_name),
            "document_name": document_name,
            "page": page,
            "snippet": snippet,
            "preview": snippet[:80],
        }
        for (document_name, page), snippet in sorted(source_map.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------

def get_last_summarized_end_message_id(long_term_memory_store, session_id: str) -> int:
    if long_term_memory_store is None or not hasattr(long_term_memory_store, "_collection"):
        return 0

    try:
        result = long_term_memory_store._collection.get(
            where={
                "$and": [
                    {"session_id": session_id},
                    {"memory_type": "summary_window"},
                ]
            },
            include=["metadatas"],
        )
    except Exception:
        logging.exception(
            "Failed to load long-term summary metadata for session_id=%s",
            session_id,
        )
        return 0

    metadatas = result.get("metadatas") or []
    end_ids = [
        int(metadata["end_message_id"])
        for metadata in metadatas
        if metadata and metadata.get("end_message_id") is not None
    ]
    return max(end_ids, default=0)


def is_window_worthy_for_long_term(messages: list[Message]) -> bool:
    if len(messages) < SHORT_TERM_MESSAGE_LIMIT:
        return False

    combined_text = "\n".join(
        message.content.strip()
        for message in messages
        if message.content.strip()
    )
    if len(combined_text) >= 120:
        return True

    high_value_keywords = (
        "我叫", "我是", "喜欢", "偏好", "目标", "计划",
        "记住", "以后", "学校", "专业", "年级", "不要",
        "必须", "约束", "总结", "结论",
    )
    lowered_text = combined_text.lower()
    return any(keyword in combined_text or keyword in lowered_text for keyword in high_value_keywords)


def summarize_window(messages: list[Message]) -> str:
    transcript = "\n".join(
        f"[{message.role}]: {message.content}"
        for message in messages
    )
    prompt = (
        "请把以下 4 轮对话压缩成适合长期记忆检索的摘要。\n"
        "只保留高价值信息：用户身份、偏好、目标、长期约束、已确认事实、重要结论。\n"
        "如果没有值得长期保留的信息，只输出 NO_MEMORY。\n\n"
        f"{transcript}"
    )
    response = get_llm_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
    )
    summary = (response.choices[0].message.content or "").strip()
    if not summary or summary == "NO_MEMORY":
        return ""
    return summary


def persist_long_term_summary(long_term_memory_store, request: ChatRequest) -> None:
    if long_term_memory_store is None:
        return
    # Write is gated by the same flag as injection: with prompt injection off,
    # summaries would be write-only (LLM + embedding cost, zero benefit).
    if not CHROMA_SESSION_SUMMARY_IN_PROMPT:
        return

    last_summarized_end_id = get_last_summarized_end_message_id(
        long_term_memory_store, request.session_id
    )

    with Session(engine) as db:
        stmt = (
            select(Message)
            .where(
                Message.session_id == request.session_id,
                Message.id > last_summarized_end_id,
            )
            .order_by(asc(Message.id))
            .limit(SHORT_TERM_MESSAGE_LIMIT)
        )
        candidate_messages = db.scalars(stmt).all()

    if len(candidate_messages) < SHORT_TERM_MESSAGE_LIMIT:
        return

    if not is_window_worthy_for_long_term(candidate_messages):
        return

    summary_text = summarize_window(candidate_messages)
    if not summary_text:
        return

    start_message_id = candidate_messages[0].id
    end_message_id = candidate_messages[-1].id
    summary_key = f"{request.session_id}:{start_message_id}:{end_message_id}"
    memory_metadata = {
        "session_id": request.session_id,
        "memory_type": "summary_window",
        "summary_key": summary_key,
        "start_message_id": start_message_id,
        "end_message_id": end_message_id,
    }

    try:
        long_term_memory_store.add_texts(
            texts=[summary_text],
            metadatas=[memory_metadata],
            ids=[str(uuid.uuid4())],
        )

        if hasattr(long_term_memory_store, "persist"):
            long_term_memory_store.persist()
    except Exception:
        logging.exception(
            "Failed to persist long-term summary for session_id=%s",
            request.session_id,
        )


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

async def assemble_context_and_memory(
    user_id: str,
    session_id: str,
    current_query: str,
    retriever,
    long_term_memory_store,
    tool_registry=None,
    web_search_enabled: bool = False,
) -> dict[str, Any]:
    all_turns = await asyncio.to_thread(load_recent_turns, session_id)
    near_turns, overflow_turns = split_near_overflow(
        all_turns,
        near_limit=SHORT_TERM_MESSAGE_LIMIT,
    )
    prev_user = previous_user_utterance(near_turns or all_turns, current_query)
    if prev_user is None:
        # Follow-up topic may sit in overflow; scan full history once.
        prev_user = previous_user_utterance(all_turns, current_query)
    resolved_query = await asyncio.to_thread(
        resolve_followup_query_llm, current_query, prev_user
    )

    def get_overflow_and_profile() -> tuple[str, str]:
        with Session(engine) as db:
            overflow_summary = load_or_refresh_overflow_summary(
                db,
                session_id=session_id,
                overflow=overflow_turns,
            )
            structured_memory = memory_manager.render_active_context(
                db,
                user_id,
                query=resolved_query or current_query,
                limit=PROFILE_INJECT_K,
            )
        return overflow_summary, structured_memory

    def get_long_term(structured_memory: str) -> str:
        if long_term_memory_store is None or not CHROMA_SESSION_SUMMARY_IN_PROMPT:
            return structured_memory

        # Optional legacy Chroma session summaries (off by default; MySQL overflow covers WM).
        docs = long_term_memory_store.similarity_search(
            query=resolved_query or current_query,
            k=1,
            filter={
                "$and": [
                    {"session_id": session_id},
                    {"memory_type": "summary_window"},
                ]
            },
        )
        summary_memory = "\n\n".join(
            doc.page_content.strip()
            for doc in docs
            if getattr(doc, "page_content", "").strip()
        )
        return "\n\n".join(item for item in (structured_memory, summary_memory) if item)

    async def get_retrieval():
        return await collect_evidence(
            resolved_query or current_query,
            retriever=retriever,
            tools=tool_registry,
            web_search_enabled=web_search_enabled,
        )

    (overflow_summary, structured_memory), retrieval = await asyncio.gather(
        asyncio.to_thread(get_overflow_and_profile),
        get_retrieval(),
    )
    long_term = await asyncio.to_thread(get_long_term, structured_memory)

    return {
        "overflow_summary": overflow_summary,
        "long_term": long_term,
        "knowledge": retrieval.knowledge_context,
        "web": format_web_evidence_for_prompt(retrieval.web_sources),
        "sources": retrieval.sources,
        "retrieval_decision": retrieval.decision,
        "recent_turns": near_turns,
        "overflow_turns": overflow_turns,
        "resolved_query": resolved_query,
        "previous_user": prev_user,
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def resolve_tool_messages(first_message: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    resolved_tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, str]] = []

    for tool_call in first_message.tool_calls:
        tool_name = tool_call.function.name
        tool_func = AVAILABLE_TOOLS_MAP.get(tool_name)
        if tool_func is None:
            logging.warning("Tool not found, skip execution: %s", tool_name)
            continue

        try:
            tool_args = json.loads(tool_call.function.arguments or "{}")
            if not isinstance(tool_args, dict):
                tool_args = {}
        except json.JSONDecodeError:
            logging.exception(
                "Invalid tool arguments, fallback to empty dict for tool=%s",
                tool_name,
            )
            tool_args = {}

        try:
            tool_result = await asyncio.to_thread(tool_func, **tool_args)
        except Exception:
            logging.exception("Tool execution failed: %s", tool_name)
            tool_result = json.dumps(
                {
                    "status": "error",
                    "message": f"工具 {tool_name} 执行失败，请稍后重试。",
                },
                ensure_ascii=False,
            )

        resolved_tool_calls.append(
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )

    return resolved_tool_calls, tool_messages


async def run_agent_loop(messages: list[dict[str, Any]]) -> str:
    first_response = await asyncio.to_thread(
        create_llm_completion,
        messages,
        with_tools=True,
        stream=False,
    )
    first_choice = first_response.choices[0]
    first_message = first_choice.message

    if (
        first_choice.finish_reason != "tool_calls"
        or not first_message.tool_calls
    ):
        return first_message.content or ""

    resolved_tool_calls, tool_messages = await resolve_tool_messages(first_message)
    if not resolved_tool_calls:
        fallback_response = await asyncio.to_thread(
            create_llm_completion,
            messages,
            with_tools=False,
            stream=False,
        )
        return fallback_response.choices[0].message.content or ""

    followup_messages = messages + [
        {
            "role": first_message.role,
            "content": first_message.content or "",
            "tool_calls": resolved_tool_calls,
        }
    ] + tool_messages

    final_response = await asyncio.to_thread(
        create_llm_completion,
        followup_messages,
        with_tools=False,
        stream=False,
    )
    return final_response.choices[0].message.content or ""


def _sse_pack(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_text_events(
    messages: list[dict[str, Any]],
    *,
    generation_options: dict[str, float],
    emit: Callable[[dict[str, Any]], None],
) -> str:
    reply_parts: list[str] = []
    first_token = True
    async for token in stream_llm_text(messages, **generation_options):
        if first_token:
            emit({"type": "start"})
            first_token = False
        reply_parts.append(token)
        emit({"type": "delta", "content": token})

    if not reply_parts:
        emit({"type": "start"})

    return "".join(reply_parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def build_retrieval_refusal(retrieval_decision: str) -> str:
    if retrieval_decision == "insufficient":
        return "当前知识库检索到了相关片段，但证据不足以支持可靠回答。建议开启联网搜索后重试；网络结果仅供参考，请以学校官网和正式文件为准。"
    return "当前知识库未收录可核验的相关依据。建议开启联网搜索后重试；网络结果仅供参考，请以学校官网和正式文件为准。"


@router.post("/chat")
async def chat(
    request_body: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    request_body.user_id = current_user.id
    request.state.session_id = request_body.session_id
    bind_log_context(user_id=current_user.id, session_id=request_body.session_id)
    app_state = request.app.state
    await asyncio.to_thread(ensure_session_exists, request_body)
    hybrid_context = await assemble_context_and_memory(
        user_id=request_body.user_id,
        session_id=request_body.session_id,
        current_query=request_body.new_message,
        retriever=getattr(app_state, "retriever", None),
        long_term_memory_store=getattr(app_state, "long_term_memory_store", None),
        tool_registry=getattr(app_state, "tool_registry", None),
        web_search_enabled=request_body.web_search_enabled,
    )
    retrieval_decision = hybrid_context["retrieval_decision"]
    if retrieval_decision in {"out_of_scope", "insufficient"}:
        ai_reply = build_retrieval_refusal(retrieval_decision)
    else:
        final_messages = build_final_messages(request_body, hybrid_context)
        ai_reply = await run_agent_loop(final_messages)
    user_message_id = await asyncio.to_thread(save_messages, request_body, ai_reply)
    memory_actions = await asyncio.to_thread(
        persist_explicit_memories,
        request_body.user_id,
        request_body.session_id,
        request_body.new_message,
        user_message_id,
    )
    await asyncio.to_thread(
        persist_long_term_summary,
        getattr(app_state, "long_term_memory_store", None),
        request_body,
    )
    confidence = calculate_confidence(hybrid_context.get("sources", []))
    return {
        "reply": ai_reply,
        "sources": [source.to_dict() for source in hybrid_context.get("sources", [])],
        "confidence": confidence.score,
        "confidence_level": confidence.level,
        "evidence_summary": confidence.evidence_summary,
        "uncertain_points": confidence.uncertain_points,
        "retrieval_decision": hybrid_context["retrieval_decision"],
        "memory_actions": memory_actions,
    }


@router.post("/chat/stream")
async def chat_stream(
    request_body: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream chat events over SSE using the same Authorization / cookie auth as REST."""
    request_body.user_id = current_user.id
    request.state.session_id = request_body.session_id
    context_tokens = bind_log_context(
        request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
        user_id=current_user.id,
        session_id=request_body.session_id,
    )
    app_state = request.app.state

    async def event_generator() -> AsyncIterator[str]:
        queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()

        def emit(payload: dict[str, Any]) -> None:
            queue.put_nowait(payload)

        async def run_pipeline() -> None:
            try:
                await asyncio.to_thread(ensure_session_exists, request_body)
                generation_options = await asyncio.to_thread(
                    get_generation_options,
                    request_body.user_id,
                )

                emit({
                    "type": "thinking",
                    "step": "analyzing",
                    "message": "正在分析你的问题",
                })

                recent_turns = await asyncio.to_thread(
                    load_recent_turns,
                    request_body.session_id,
                )
                previous_user = previous_user_utterance(
                    recent_turns,
                    request_body.new_message,
                )
                intent = await classify_intent(
                    request_body.new_message,
                    previous_user=previous_user,
                )

                if intent == "chat":
                    emit({
                        "type": "thinking",
                        "step": "generating",
                        "message": "正在生成回答",
                    })
                    chat_messages = build_chat_messages(
                        request_body.new_message,
                        recent_turns=recent_turns,
                    )
                    ai_reply = await _stream_text_events(
                        chat_messages,
                        generation_options=generation_options,
                        emit=emit,
                    )
                    user_message_id = await asyncio.to_thread(
                        save_messages,
                        request_body,
                        ai_reply,
                    )
                    memory_actions = await asyncio.to_thread(
                        persist_explicit_memories,
                        request_body.user_id,
                        request_body.session_id,
                        request_body.new_message,
                        user_message_id,
                    )
                    emit({"type": "end", "sources": [], "memory_actions": memory_actions})
                    return

                emit({
                    "type": "thinking",
                    "step": "knowledge_retrieval",
                    "message": "正在检索校内知识库",
                })
                if request_body.web_search_enabled:
                    emit({
                        "type": "thinking",
                        "step": "web_search",
                        "message": "正在通过联网搜索工具查询互联网",
                    })

                hybrid_context = await assemble_context_and_memory(
                    user_id=request_body.user_id,
                    session_id=request_body.session_id,
                    current_query=request_body.new_message,
                    retriever=getattr(app_state, "retriever", None),
                    long_term_memory_store=getattr(app_state, "long_term_memory_store", None),
                    tool_registry=getattr(app_state, "tool_registry", None),
                    web_search_enabled=request_body.web_search_enabled,
                )

                if hybrid_context["retrieval_decision"] in {"out_of_scope", "insufficient"}:
                    ai_reply = build_retrieval_refusal(hybrid_context["retrieval_decision"])
                    emit({"type": "start"})
                    emit({"type": "delta", "content": ai_reply})
                    user_message_id = await asyncio.to_thread(
                        save_messages,
                        request_body,
                        ai_reply,
                    )
                    memory_actions = await asyncio.to_thread(
                        persist_explicit_memories,
                        request_body.user_id,
                        request_body.session_id,
                        request_body.new_message,
                        user_message_id,
                    )
                    emit({
                        "type": "end",
                        "sources": [],
                        "retrieval_decision": hybrid_context["retrieval_decision"],
                        "memory_actions": memory_actions,
                    })
                    return

                emit({
                    "type": "thinking",
                    "step": "evidence_fusion",
                    "message": "正在整理可核验的参考资料",
                    "detail": (
                        "已获取相关参考资料"
                        if hybrid_context.get("sources")
                        else "未检索到相关内容"
                    ),
                })
                emit({
                    "type": "thinking",
                    "step": "generating",
                    "message": "正在生成回答",
                })

                final_messages = build_final_messages(request_body, hybrid_context)
                ai_reply = await _stream_text_events(
                    final_messages,
                    generation_options=generation_options,
                    emit=emit,
                )
                user_message_id = await asyncio.to_thread(
                    save_messages,
                    request_body,
                    ai_reply,
                )
                memory_actions = await asyncio.to_thread(
                    persist_explicit_memories,
                    request_body.user_id,
                    request_body.session_id,
                    request_body.new_message,
                    user_message_id,
                )
                await asyncio.to_thread(
                    persist_long_term_summary,
                    getattr(app_state, "long_term_memory_store", None),
                    request_body,
                )

                confidence = calculate_confidence(hybrid_context.get("sources", []))
                emit({
                    "type": "end",
                    "sources": [
                        source.to_dict()
                        for source in hybrid_context.get("sources", [])
                    ],
                    "confidence": confidence.score,
                    "confidence_level": confidence.level,
                    "evidence_summary": confidence.evidence_summary,
                    "uncertain_points": confidence.uncertain_points,
                    "retrieval_decision": hybrid_context["retrieval_decision"],
                    "memory_actions": memory_actions,
                })
            except HTTPException as exc:
                detail = exc.detail
                if not isinstance(detail, str):
                    detail = "请求失败"
                emit({"type": "error", "message": detail})
            except Exception:
                logging.exception("SSE chat failed")
                emit({"type": "error", "message": "Agent 处理失败，请稍后重试。"})
            finally:
                queue.put_nowait(None)

        worker = asyncio.create_task(run_pipeline())
        try:
            while True:
                if await request.is_disconnected():
                    worker.cancel()
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    break
                yield _sse_pack(item)
        finally:
            if not worker.done():
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
            reset_log_context(context_tokens)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
