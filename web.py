import uuid

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"
WELCOME_MESSAGE = (
    "你好！我是重邮极客 Agent。"
    "我可以帮你查询天气、课表，也可以检索和解答重邮学生手册相关问题。"
)


st.set_page_config(
    page_title="重邮极客 Agent",
    page_icon="💬",
    layout="centered",
)


# 用少量样式把页面收得更像极简聊天界面。
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
    """为当前浏览器会话生成一个简短且可读的用户标识。"""
    return f"user_{uuid.uuid4().hex[:4]}"


def build_session_id() -> str:
    """生成满足后端长度要求的新会话标识。"""
    return f"session_{uuid.uuid4().hex[:12]}"


def init_chat_state() -> None:
    """初始化页面级会话状态，只在首次打开页面时执行一次。"""
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

    # 兼容旧版本页面里只提天气/课表的欢迎语，自动替换成新的能力说明。
    if (
        st.session_state.messages
        and st.session_state.messages[0].get("role") == "assistant"
        and (
            "天气或者课程安排" in st.session_state.messages[0].get("content", "")
            or "天气、课表之类的问题" in st.session_state.messages[0].get("content", "")
        )
    ):
        st.session_state.messages[0]["content"] = WELCOME_MESSAGE


def start_new_chat() -> None:
    """新建对话时只刷新会话标识，并清空当前聊天记录。"""
    st.session_state.session_id = build_session_id()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]


init_chat_state()


with st.sidebar:
    # 用大按钮作为唯一主操作，模拟 ChatGPT 的新建对话体验。
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
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
            <p>你可以问天气、课表，也可以查询重邮学生手册相关内容。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 用 session_state 保持多轮聊天记录。
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("输入消息，开始一段新对话...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = None
        reply = ""

        try:
            # 请求后端时自动拼装当前浏览器对应的 user_id 和 session_id。
            with st.spinner("Agent 正在思考调工具..."):
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
            st.markdown(reply)

        except requests.RequestException as exc:
            error_message = "接口调用失败，请检查后端服务是否启动。"

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

            st.error(f"{error_message}\n\n{exc}")

        except ValueError:
            st.error("后端返回的不是合法 JSON，无法解析响应内容。")

    if reply:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )
