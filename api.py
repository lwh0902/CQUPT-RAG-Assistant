import asyncio
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field, ValidationError, field_validator
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
    title="重邮极客 AI 大脑",
    version="1.1.0",
)

load_dotenv()

api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("未找到 ZHIPU_API_KEY，请检查 .env 配置。")

client = ZhipuAI(api_key=api_key)

SHORT_TERM_ROUNDS = 4
MESSAGES_PER_ROUND = 2
SHORT_TERM_MESSAGE_LIMIT = SHORT_TERM_ROUNDS * MESSAGES_PER_ROUND
LONG_TERM_MEMORY_DIR = Path("./chroma_memory_db")
LONG_TERM_MEMORY_COLLECTION = "chat_long_term_memory"

retriever = None
retriever_init_info: dict[str, object] = {}
long_term_memory_store = None


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
        min_length=1,
        max_length=2000,
        description="用户最新发送的聊天内容",
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


def build_system_prompt(hybrid_context: dict[str, str]) -> str:
    knowledge = hybrid_context.get("knowledge") or "无"
    long_term = hybrid_context.get("long_term") or "无"
    short_term = hybrid_context.get("short_term") or "无"

    return (
        "你是“重邮极客 Agent”，是一个校园综合助手。\n"
        "你的能力包括：\n"
        "1. 基于学生手册知识回答校内制度、部门、办事流程、学籍、奖助、纪律等问题；\n"
        "2. 在需要时调用工具查询天气；\n"
        "3. 在需要时调用工具查询课表。\n"
        "当用户打招呼、寒暄，或者询问“你能做什么”时，"
        "请明确告诉用户你既可以查天气和课表，也可以检索并解答学生手册相关内容。\n"
        "请优先基于以下参考资料回答问题；如果问题需要实时天气或课表信息，就结合工具结果回答。\n\n"
        f"【学生手册知识】\n{knowledge}\n\n"
        f"【长期记忆】\n{long_term}\n\n"
        f"【短期记忆】\n{short_term}"
    )


def build_final_messages(
    request: ChatRequest,
    hybrid_context: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_system_prompt(hybrid_context)},
        {"role": "user", "content": request.new_message},
    ]


def create_glm_completion(
    messages: list[dict[str, Any]],
    *,
    with_tools: bool,
    stream: bool = False,
) -> Any:
    request_payload: dict[str, Any] = {
        "model": "glm-4",
        "messages": messages,
        "stream": stream,
    }
    if with_tools:
        request_payload["tools"] = AGENT_TOOLS_SCHEMA
        request_payload["tool_choice"] = "auto"
    return client.chat.completions.create(**request_payload)


def ensure_session_exists(request: ChatRequest) -> None:
    with Session(engine) as db:
        user_stmt = select(User).where(User.id == request.user_id)
        current_user = db.scalar(user_stmt)

        if current_user is None:
            current_user = User(
                id=request.user_id,
                username=f"user_{request.user_id}",
            )
            db.add(current_user)
            db.flush()

        session_stmt = select(ChatSession).where(ChatSession.id == request.session_id)
        chat_window = db.scalar(session_stmt)

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

        if chat_window.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="无权访问该聊天会话。")


def save_messages(request: ChatRequest, ai_reply: str) -> None:
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


def get_last_summarized_end_message_id(session_id: str) -> int:
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
    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": prompt}],
    )
    summary = (response.choices[0].message.content or "").strip()
    if not summary or summary == "NO_MEMORY":
        return ""
    return summary


def persist_long_term_summary(request: ChatRequest) -> None:
    if long_term_memory_store is None:
        return

    last_summarized_end_id = get_last_summarized_end_message_id(request.session_id)

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
        create_glm_completion,
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
            create_glm_completion,
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
        create_glm_completion,
        followup_messages,
        with_tools=False,
        stream=False,
    )
    return final_response.choices[0].message.content or ""


async def stream_glm_text(messages: list[dict[str, Any]]):
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def put_event(event_type: str, payload: str = "") -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

    def worker() -> None:
        try:
            stream_response = create_glm_completion(
                messages,
                with_tools=False,
                stream=True,
            )
            for chunk in stream_response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    put_event("delta", content)
        except Exception as exc:
            logging.exception("Streaming completion failed")
            put_event("error", str(exc))
        finally:
            put_event("done")

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event_type, payload = await queue.get()
        if event_type == "delta":
            yield payload
            continue
        if event_type == "error":
            raise RuntimeError(payload)
        break


async def run_agent_loop_stream(
    messages: list[dict[str, Any]],
    websocket: WebSocket,
) -> str:
    first_response = await asyncio.to_thread(
        create_glm_completion,
        messages,
        with_tools=True,
        stream=False,
    )
    first_choice = first_response.choices[0]
    first_message = first_choice.message

    target_messages = messages
    if (
        first_choice.finish_reason == "tool_calls"
        and first_message.tool_calls
    ):
        resolved_tool_calls, tool_messages = await resolve_tool_messages(first_message)
        if resolved_tool_calls:
            target_messages = messages + [
                {
                    "role": first_message.role,
                    "content": first_message.content or "",
                    "tool_calls": resolved_tool_calls,
                }
            ] + tool_messages

    await websocket.send_json({"type": "start"})

    reply_parts: list[str] = []
    async for token in stream_glm_text(target_messages):
        reply_parts.append(token)
        await websocket.send_json(
            {
                "type": "delta",
                "content": token,
            }
        )

    return "".join(reply_parts)


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
            "message": "参数校验失败，请检查输入规范。",
            "details": details,
        },
    )


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "欢迎来到重邮极客 Agent 服务。"}


@app.on_event("startup")
async def startup_event() -> None:
    def init_long_term_memory_store() -> Chroma:
        LONG_TERM_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(LONG_TERM_MEMORY_DIR),
            collection_name=LONG_TERM_MEMORY_COLLECTION,
            embedding_function=get_embeddings(),
        )

    global retriever, retriever_init_info, long_term_memory_store
    retriever_data, long_term_store = await asyncio.gather(
        asyncio.to_thread(init_rag_system),
        asyncio.to_thread(init_long_term_memory_store),
    )
    retriever, retriever_init_info = retriever_data
    long_term_memory_store = long_term_store


async def assemble_context_and_memory(session_id: str, current_query: str) -> dict[str, str]:
    def get_short_term() -> str:
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

    def get_long_term() -> str:
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

    async def get_knowledge() -> str:
        if retriever is None:
            return ""
        return await get_rag_context_async(current_query, retriever)

    short_term, long_term, knowledge = await asyncio.gather(
        asyncio.to_thread(get_short_term),
        asyncio.to_thread(get_long_term),
        get_knowledge(),
    )

    return {
        "short_term": short_term,
        "long_term": long_term,
        "knowledge": knowledge,
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    await asyncio.to_thread(ensure_session_exists, request)
    hybrid_context = await assemble_context_and_memory(
        session_id=request.session_id,
        current_query=request.new_message,
    )
    final_messages = build_final_messages(request, hybrid_context)
    ai_reply = await run_agent_loop(final_messages)
    await asyncio.to_thread(save_messages, request, ai_reply)
    await asyncio.to_thread(persist_long_term_summary, request)
    return {"reply": ai_reply}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        payload = await websocket.receive_json()
        request = ChatRequest.model_validate(payload)
    except ValidationError as exc:
        details = [
            {
                "loc": " -> ".join(str(item) for item in error.get("loc", [])),
                "msg": error.get("msg", ""),
            }
            for error in exc.errors()
        ]
        await websocket.send_json(
            {
                "type": "error",
                "message": "参数校验失败，请检查输入规范。",
                "details": details,
            }
        )
        await websocket.close(code=1008)
        return
    except Exception as exc:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"WebSocket 初始化失败：{exc}",
            }
        )
        await websocket.close(code=1003)
        return

    try:
        await asyncio.to_thread(ensure_session_exists, request)
        hybrid_context = await assemble_context_and_memory(
            session_id=request.session_id,
            current_query=request.new_message,
        )
        final_messages = build_final_messages(request, hybrid_context)
        ai_reply = await run_agent_loop_stream(final_messages, websocket)
        await asyncio.to_thread(save_messages, request, ai_reply)
        await asyncio.to_thread(persist_long_term_summary, request)
        await websocket.send_json({"type": "end"})
    except HTTPException as exc:
        await websocket.send_json(
            {
                "type": "error",
                "message": exc.detail,
            }
        )
    except Exception as exc:
        logging.exception("WebSocket chat failed")
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Agent 处理失败：{exc}",
            }
        )
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(
        app="api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
