from __future__ import annotations

import json
import os
import uuid
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
import requests
import streamlit as st


load_dotenv()

API_TIMEOUT = 60
HISTORY_TIMEOUT = 20
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
WELCOME_MESSAGE = (
    "你好，我是重邮极客 Agent。\n"
    "我可以帮你查询天气、课表，也可以检索和解答学生手册相关问题。"
)


def normalize_base_url(url: str) -> str:
    return (url or DEFAULT_API_BASE_URL).rstrip("/")


def derive_websocket_base(api_base_url: str) -> str:
    parsed = urlparse(api_base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "", "", "", "")).rstrip("/")


API_BASE_URL = normalize_base_url(
    os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
)
WS_BASE_URL = normalize_base_url(
    os.getenv("WS_BASE_URL", derive_websocket_base(API_BASE_URL))
)
CHAT_API_URL = f"{API_BASE_URL}/chat"
SESSIONS_API_URL = f"{API_BASE_URL}/sessions"
WS_CHAT_URL = f"{WS_BASE_URL}/ws/chat"


st.set_page_config(
    page_title="重邮极客 Agent",
    page_icon="🤖",
    layout="centered",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top, rgba(255, 255, 255, 0.96), rgba(244, 244, 245, 0.96)),
            linear-gradient(135deg, #eef6ff 0%, #f8f3ea 100%);
    }
    .block-container {
        max-width: 920px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.92);
        border-right: 1px solid #e5e7eb;
    }
    .debug-meta {
        color: #6b7280;
        font-size: 0.78rem;
        line-height: 1.7;
        word-break: break-all;
    }
    .sidebar-note {
        color: #4b5563;
        font-size: 0.82rem;
        line-height: 1.6;
        margin-bottom: 0.9rem;
    }
    .session-preview {
        color: #6b7280;
        font-size: 0.76rem;
        line-height: 1.55;
        margin: -0.35rem 0 0.8rem 0.2rem;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        color: #111827;
        font-size: 2.1rem;
    }
    .hero p {
        margin: 0;
        color: #4b5563;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


def build_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


def normalize_message_role(role: str) -> str:
    return "assistant" if role in {"assistant", "ai"} else role


def default_messages() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def init_chat_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = build_user_id()

    if "session_id" not in st.session_state:
        st.session_state.session_id = build_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = default_messages()

    if "session_summaries" not in st.session_state:
        st.session_state.session_summaries = []

    if "history_error" not in st.session_state:
        st.session_state.history_error = ""


def request_json(method: str, url: str, **kwargs) -> dict:
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


def extract_error_message(response: requests.Response | None, fallback: str) -> str:
    if response is None:
        return fallback

    try:
        error_body = response.json()
    except ValueError:
        return fallback

    detail = error_body.get("detail")
    if isinstance(detail, list):
        detail = "；".join(
            item.get("msg", "")
            for item in detail
            if isinstance(item, dict)
        )

    return error_body.get("message") or detail or fallback


def refresh_session_summaries() -> None:
    try:
        data = request_json(
            "GET",
            SESSIONS_API_URL,
            params={"user_id": st.session_state.user_id},
            timeout=HISTORY_TIMEOUT,
        )
        st.session_state.session_summaries = data.get("sessions", [])
        st.session_state.history_error = ""
    except requests.RequestException:
        st.session_state.session_summaries = []
        st.session_state.history_error = "历史会话暂时无法加载，请先确认后端服务已启动。"
    except ValueError:
        st.session_state.session_summaries = []
        st.session_state.history_error = "历史会话接口返回的不是合法 JSON。"


def load_existing_session(session_id: str) -> None:
    data = request_json(
        "GET",
        f"{SESSIONS_API_URL}/{session_id}/messages",
        params={"user_id": st.session_state.user_id},
        timeout=HISTORY_TIMEOUT,
    )
    messages = data.get("messages", [])
    st.session_state.session_id = session_id
    st.session_state.messages = [
        {
            "role": normalize_message_role(message.get("role", "assistant")),
            "content": message.get("content", ""),
        }
        for message in messages
        if message.get("content")
    ] or default_messages()


def start_new_chat() -> None:
    st.session_state.session_id = build_session_id()
    st.session_state.messages = default_messages()


def stream_reply_via_websocket(prompt: str, response_placeholder) -> str:
    import websocket

    ws = None
    reply = ""

    try:
        ws = websocket.create_connection(WS_CHAT_URL, timeout=API_TIMEOUT)
        ws.send(
            json.dumps(
                {
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "new_message": prompt,
                },
                ensure_ascii=False,
            )
        )

        while True:
            raw_message = ws.recv()
            event = json.loads(raw_message)
            event_type = event.get("type")

            if event_type == "delta":
                reply += event.get("content", "")
                response_placeholder.markdown(reply)
                continue

            if event_type == "error":
                raise RuntimeError(event.get("message", "WebSocket 调用失败"))

            if event_type == "end":
                break
    finally:
        if ws is not None:
            ws.close()

    return reply


init_chat_state()
refresh_session_summaries()


with st.sidebar:
    if st.button("新建对话", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()
    st.markdown("### 历史会话")
    st.markdown(
        "<div class='sidebar-note'>这里会展示当前用户已经保存到数据库的聊天记录。</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.history_error:
        st.caption(st.session_state.history_error)
    elif not st.session_state.session_summaries:
        st.caption("暂无历史会话，先问一个问题试试。")
    else:
        for session in st.session_state.session_summaries:
            is_active = session["session_id"] == st.session_state.session_id
            title = session.get("title") or "未命名对话"
            count = session.get("message_count", 0)
            label = f"{title} ({count}条)" if count else title

            if st.button(
                label,
                key=f"session_{session['session_id']}",
                use_container_width=True,
                disabled=is_active,
            ):
                try:
                    load_existing_session(session["session_id"])
                    refresh_session_summaries()
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"加载会话失败：{exc}")
                except ValueError:
                    st.error("历史消息接口返回的不是合法 JSON。")

            preview = session.get("preview")
            if preview:
                st.markdown(
                    f"<div class='session-preview'>{preview}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown("### 系统信息")
    st.markdown(
        (
            "<div class='debug-meta'>"
            f"user_id: {st.session_state.user_id}<br>"
            f"session_id: {st.session_state.session_id}<br>"
            f"api: {API_BASE_URL}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>校园知识库智能问答系统</h1>
        <p>围绕学生手册问答、历史会话管理和工具查询构建的课程项目原型。现在你可以直接切换旧会话继续问，也可以新开一段全新的对话。</p>
    </div>
    """,
    unsafe_allow_html=True,
)


for message in st.session_state.messages:
    with st.chat_message(normalize_message_role(message["role"])):
        st.markdown(message["content"])


prompt = st.chat_input("输入消息，开始一段新对话...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply = ""
        response_placeholder = st.empty()
        used_websocket = False

        try:
            reply = stream_reply_via_websocket(prompt, response_placeholder)
            used_websocket = True
        except Exception as ws_exc:
            if used_websocket or reply:
                st.error(str(ws_exc))
            else:
                response = None
                try:
                    with st.spinner("Agent 正在思考并调工具..."):
                        response = requests.post(
                            CHAT_API_URL,
                            json={
                                "user_id": st.session_state.user_id,
                                "session_id": st.session_state.session_id,
                                "new_message": prompt,
                            },
                            timeout=API_TIMEOUT,
                        )

                    response.raise_for_status()
                    data = response.json()
                    reply = data.get("reply", "后端没有返回 reply 字段。")
                    response_placeholder.markdown(reply)
                except requests.RequestException as http_exc:
                    error_message = extract_error_message(
                        response,
                        "接口调用失败，请检查后端服务是否已启动。",
                    )
                    st.error(f"{error_message}\n\n{http_exc}")
                except ValueError:
                    st.error("后端返回的不是合法 JSON，无法解析响应内容。")

    if reply:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )
        refresh_session_summaries()
