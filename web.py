import json
import uuid

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"
WS_URL = "ws://127.0.0.1:8000/ws/chat"
WELCOME_MESSAGE = (
    "你好，我是重邮极客 Agent。\n"
    "我可以帮你查天气、课表，也可以检索和解答学生手册相关问题。"
)


st.set_page_config(
    page_title="重邮极客 Agent",
    page_icon="🤖",
    layout="centered",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f7f7f8;
    }
    .block-container {
        max-width: 880px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #ececf1;
    }
    .debug-meta {
        color: #8e8ea0;
        font-size: 0.78rem;
        line-height: 1.7;
        word-break: break-all;
    }
    .empty-state {
        padding: 3.5rem 0 2rem 0;
        color: #6e6e80;
        text-align: center;
    }
    .empty-state h1 {
        color: #202123;
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:4]}"


def build_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


def init_chat_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = build_user_id()

    if "session_id" not in st.session_state:
        st.session_state.session_id = build_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]


def start_new_chat() -> None:
    st.session_state.session_id = build_session_id()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]


def stream_reply_via_websocket(prompt: str, response_placeholder) -> str:
    import websocket

    ws = None
    reply = ""

    try:
        ws = websocket.create_connection(WS_URL, timeout=60)
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


with st.sidebar:
    if st.button("↻ 新建对话", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()
    st.markdown("### 重邮极客 Agent")
    st.caption("天气、课表、学生手册问答都可以直接聊。")
    st.markdown(
        (
            "<div class='debug-meta'>"
            f"user_id: {st.session_state.user_id}<br>"
            f"session_id: {st.session_state.session_id}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


st.title("重邮极客 Agent")
st.caption("像聊天一样提问，系统会自动维护身份与会话。")


if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <h1>今天想问点什么？</h1>
            <p>你可以问天气、课表，也可以查询学生手册相关内容。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
                            API_URL,
                            json={
                                "user_id": st.session_state.user_id,
                                "session_id": st.session_state.session_id,
                                "new_message": prompt,
                            },
                            timeout=60,
                        )

                    response.raise_for_status()
                    data = response.json()
                    reply = data.get("reply", "后端没有返回 reply 字段。")
                    response_placeholder.markdown(reply)
                except requests.RequestException as http_exc:
                    error_message = "接口调用失败，请检查后端服务是否已启动。"

                    try:
                        if response is not None:
                            error_body = response.json()
                            detail = error_body.get("detail")
                            if isinstance(detail, list):
                                detail = "；".join(
                                    item.get("msg", "")
                                    for item in detail
                                    if isinstance(item, dict)
                                )
                            error_message = error_body.get("message") or detail or error_message
                    except Exception:
                        pass

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
