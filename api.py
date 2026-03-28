import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session
from zhipuai import ZhipuAI
import uvicorn

from database import engine
from models import ChatSession, Message, User
from rag import get_rag_context_async, init_rag_system
from tools import AGENT_TOOLS_SCHEMA, AVAILABLE_TOOLS_MAP
from vector_store import get_embeddings


app = FastAPI(
    title="重邮极客的超级 AI 大脑",
    version="1.0.0",
)

load_dotenv()

api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("未找到 ZHIPU_API_KEY，请检查 .env 配置。")

client = ZhipuAI(api_key=api_key)

SHORT_TERM_ROUNDS = 4
MESSAGES_PER_ROUND = 2
SHORT_TERM_MESSAGE_LIMIT = SHORT_TERM_ROUNDS * MESSAGES_PER_ROUND
retriever = None
retriever_init_info: dict[str, object] = {}
long_term_memory_store = None
LONG_TERM_MEMORY_DIR = Path("./chroma_memory_db")
LONG_TERM_MEMORY_COLLECTION = "chat_long_term_memory"


class ChatRequest(BaseModel):
    user_id: str = Field(
        ...,
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
        description="用户最新发送的聊天内容",
        min_length=1,
        max_length=2000,
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
            raise ValueError("检测到非法指令：禁止 Prompt 注入！")
        return value


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    client_ip = request.client.host if request.client else "unknown"
    errors = exc.errors()
    logging.warning("Validation failed from ip=%s details=%s", client_ip, errors)

    details = [
        {
            "loc": " -> ".join(str(item) for item in error.get("loc", [])),
            "msg": error.get("msg", ""),
        }
        for error in errors
    ]

    return JSONResponse(
        status_code=422,
        content={
            "error_code": "RAG_422",
            "message": "参数校验失败，请检查输入规范",
            "details": details,
        },
    )


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "砰！欢迎来到重邮 RAG 记忆中枢！大门已敞开！"}


@app.on_event("startup")
async def startup_event() -> None:
    """服务启动时同时预热知识库检索器和长期记忆向量库实例。"""

    def _init_long_term_memory_store() -> Chroma:
        LONG_TERM_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(LONG_TERM_MEMORY_DIR),
            collection_name=LONG_TERM_MEMORY_COLLECTION,
            embedding_function=get_embeddings(),
        )

    global retriever, retriever_init_info, long_term_memory_store
    (retriever_data, long_term_store) = await asyncio.gather(
        asyncio.to_thread(init_rag_system),
        asyncio.to_thread(_init_long_term_memory_store),
    )
    retriever, retriever_init_info = retriever_data
    long_term_memory_store = long_term_store


async def assemble_context_and_memory(session_id: str, current_query: str) -> dict:
    """
    并发组装三类上下文：
    1. short_term：MySQL 中当前会话最近 4 条短期记忆
    2. long_term：Chroma 中带 session_id 元数据过滤的长期记忆
    3. knowledge：学生手册知识库检索结果
    """

    def _get_short_term() -> str:
        """从 MySQL 中提取当前会话最近 4 轮消息，作为短期工作记忆。"""
        with Session(engine) as db:
            stmt = (
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(desc(Message.id))
                .limit(SHORT_TERM_MESSAGE_LIMIT)
            )
            messages = list(reversed(db.scalars(stmt).all()))

        if not messages:
            return ""

        return "\n".join(f"[{message.role}]: {message.content}" for message in messages)

    def _get_long_term() -> str:
        """
        从长期记忆向量库中按当前问题检索历史记忆。
        这里必须加 metadata filter={"session_id": session_id}，
        作用是只允许命中当前会话自己的长期记忆，防止跨会话数据串线和越权读取。
        """
        if long_term_memory_store is None:
            return ""

        docs = long_term_memory_store.similarity_search(
            query=current_query,
            k=4,
            filter={
                "$and": [
                    {"session_id": session_id},
                    {"memory_type": "summary_window"},
                ]
            },
        )
        if not docs:
            return ""

        return "\n\n".join(
            doc.page_content.strip()
            for doc in docs
            if getattr(doc, "page_content", "").strip()
        )

    async def _get_knowledge() -> str:
        """从学生手册知识库中提取与当前问题相关的公开知识上下文。"""
        if retriever is None:
            return ""
        return await get_rag_context_async(current_query, retriever)

    short_term, long_term, knowledge = await asyncio.gather(
        asyncio.to_thread(_get_short_term),
        asyncio.to_thread(_get_long_term),
        _get_knowledge(),
    )

    return {
        "short_term": short_term,
        "long_term": long_term,
        "knowledge": knowledge,
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    def _ensure_session_exists() -> None:
        """第一次短连接：先查人建人，再查房建房，并校验会话归属。"""
        with Session(engine) as db:
            # 先按 user_id 查用户，不存在则自动注册。
            user_stmt = select(User).where(User.id == request.user_id)
            current_user = db.scalar(user_stmt)

            if current_user is None:
                current_user = User(
                    id=request.user_id,
                    username=f"user_{request.user_id}",
                )
                db.add(current_user)
                db.flush()

            # 再按 session_id 查会话。
            session_stmt = select(ChatSession).where(ChatSession.id == request.session_id)
            chat_window = db.scalar(session_stmt)

            # 会话不存在时，绑定到当前真实用户，而不是硬编码默认用户。
            if chat_window is None:
                chat_window = ChatSession(
                    id=request.session_id,
                    title="新建会话",
                    user_id=current_user.id,
                )
                db.add(chat_window)
                db.commit()
                db.refresh(chat_window)
                return

            # 会话存在但不属于当前用户时，直接拒绝访问，防止越权。
            if chat_window.user_id != current_user.id:
                raise HTTPException(status_code=403, detail="无权访问该聊天会话")

    def _create_glm_completion(
        messages: list[dict[str, Any]],
        *,
        with_tools: bool,
    ) -> Any:
        """在线程池中执行同步的大模型网络请求。"""
        request_payload: dict[str, Any] = {
            "model": "glm-4",
            "messages": messages,
        }
        if with_tools:
            request_payload["tools"] = AGENT_TOOLS_SCHEMA
            request_payload["tool_choice"] = "auto"
        return client.chat.completions.create(**request_payload)

    async def _run_agent_loop(messages: list[dict[str, Any]]) -> str:
        first_response = await asyncio.to_thread(
            _create_glm_completion,
            messages,
            with_tools=True,
        )
        first_choice = first_response.choices[0]
        first_message = first_choice.message

        if (
            first_choice.finish_reason != "tool_calls"
            or not first_message.tool_calls
        ):
            return first_message.content or ""

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

        if not resolved_tool_calls:
            fallback_response = await asyncio.to_thread(
                _create_glm_completion,
                messages,
                with_tools=False,
            )
            return fallback_response.choices[0].message.content or ""

        messages.append(
            {
                "role": first_message.role,
                "content": first_message.content or "",
                "tool_calls": resolved_tool_calls,
            }
        )
        messages.extend(tool_messages)

        final_response = await asyncio.to_thread(
            _create_glm_completion,
            messages,
            with_tools=False,
        )
        return final_response.choices[0].message.content or ""

    def _save_messages(ai_reply: str) -> None:
        """第二次短连接：将本轮用户消息和 AI 回复写回数据库。"""
        with Session(engine) as db:
            user_message = Message(
                role="user",
                content=request.new_message,
                session_id=request.session_id,
            )
            ai_message = Message(
                role="ai",
                content=ai_reply,
                session_id=request.session_id,
            )
            db.add_all([user_message, ai_message])
            db.commit()

    def _get_last_summarized_end_message_id() -> int:
        """
        从长期记忆库里找出当前会话最后一次已经摘要到哪条消息。
        这样可以避免对同一批对话窗口重复做摘要。
        """
        if long_term_memory_store is None or not hasattr(long_term_memory_store, "_collection"):
            return 0

        try:
            result = long_term_memory_store._collection.get(
                where={
                    "$and": [
                        {"session_id": request.session_id},
                        {"memory_type": "summary_window"},
                    ]
                },
                include=["metadatas"],
            )
        except Exception:
            logging.exception(
                "Failed to load long-term summary metadata for session_id=%s",
                request.session_id,
            )
            return 0

        metadatas = result.get("metadatas") or []
        end_ids = [
            int(metadata["end_message_id"])
            for metadata in metadatas
            if metadata and metadata.get("end_message_id") is not None
        ]
        return max(end_ids, default=0)

    def _is_window_worthy_for_long_term(messages: list[Message]) -> bool:
        """
        判断这 4 轮窗口是否值得沉淀到长期记忆。
        这里优先保留：身份、偏好、目标、约束、承诺、重要结论等高价值信息；
        过短寒暄、纯礼貌往返则不进入长期记忆。
        """
        if len(messages) < SHORT_TERM_MESSAGE_LIMIT:
            return False

        combined_text = "\n".join(message.content.strip() for message in messages if message.content.strip())
        if len(combined_text) >= 120:
            return True

        high_value_keywords = (
            "我叫",
            "我是",
            "喜欢",
            "偏好",
            "目标",
            "计划",
            "记住",
            "以后",
            "学校",
            "专业",
            "年级",
            "不要",
            "必须",
            "约束",
            "总结",
            "结论",
        )
        lowered_text = combined_text.lower()
        return any(keyword in combined_text or keyword in lowered_text for keyword in high_value_keywords)

    def _summarize_window(messages: list[Message]) -> str:
        """
        将 4 轮窗口压缩成长期记忆摘要。
        如果这 4 轮没有值得长期保留的信息，则返回空字符串。
        """
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
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
        )
        summary = (response.choices[0].message.content or "").strip()
        if not summary or summary == "NO_MEMORY":
            return ""
        return summary

    def _persist_long_term_summary() -> None:
        """
        长期记忆沉淀策略：
        1. 只有累计满 4 轮（8 条消息）才触发摘要；
        2. 只有高价值窗口才入长期记忆；
        3. 基于 end_message_id 避免对同一批窗口重复摘要。
        """
        if long_term_memory_store is None:
            return

        last_summarized_end_id = _get_last_summarized_end_message_id()

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

        # 没有累计满 4 轮窗口，就不做摘要。
        if len(candidate_messages) < SHORT_TERM_MESSAGE_LIMIT:
            return

        if not _is_window_worthy_for_long_term(candidate_messages):
            return

        summary_text = _summarize_window(candidate_messages)
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

    await asyncio.to_thread(_ensure_session_exists)
    hybrid_context = await assemble_context_and_memory(
        session_id=request.session_id,
        current_query=request.new_message,
    )

    # 数据组装顺序：
    # 1. 将知识库、长期记忆、短期记忆统一整理进 System Prompt
    # 2. 让模型在一个总上下文里同时参考公开知识和私有记忆
    # 3. 最后只追加当前用户问题，生成本轮回答
    system_prompt = (
        "你是“重邮极客 Agent”，是一个校园综合助手。\n"
        "你的能力包括：\n"
        "1. 基于学生手册知识回答校内制度、部门、办事流程、学籍、奖助、纪律等问题；\n"
        "2. 在需要时调用工具查询天气；\n"
        "3. 在需要时调用工具查询课表。\n"
        "当用户打招呼、寒暄，或询问“你能做什么”时，请明确告诉用户："
        "你既可以查天气和课表，也可以查询并解答重邮学生手册相关内容，"
        "不要把自己的能力描述得过窄。\n"
        "请优先基于以下参考资料回答问题；如果问题需要实时天气或课表信息，就结合工具结果回答。\n\n"
        "【学生手册知识】\n"
        f"{hybrid_context['knowledge'] or '无'}\n\n"
        "【长期记忆】\n"
        f"{hybrid_context['long_term'] or '无'}\n\n"
        "【短期记忆】\n"
        f"{hybrid_context['short_term'] or '无'}"
    )
    final_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.new_message},
    ]

    ai_reply = await _run_agent_loop(final_messages)
    await asyncio.to_thread(_save_messages, ai_reply)
    await asyncio.to_thread(_persist_long_term_summary)

    return {"reply": ai_reply}


if __name__ == "__main__":
    uvicorn.run(
        app="api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
