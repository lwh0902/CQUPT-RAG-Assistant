"""
Streamlit Web 应用入口。

界面层只负责：
1. 初始化 RAG 系统
2. 接收用户问题
3. 展示回答、页码和参考片段
"""

import asyncio

import streamlit as st

from rag import ask_question_async, init_rag_system_async


st.set_page_config(
    page_title="重邮学生手册问答",
    page_icon="📘",
    layout="centered",
)


@st.cache_resource
def init_rag_system():
    """
    初始化 RAG 系统，并缓存结果。

    这样页面刷新时不会重复创建 retriever。
    """
    return asyncio.run(init_rag_system_async())


try:
    retriever, init_info = init_rag_system()
    init_error = None
except Exception as exc:
    retriever = None
    init_info = {}
    init_error = str(exc)


with st.sidebar:
    st.markdown("### 📘 重邮学生手册问答")
    st.markdown("当前知识来源固定为《学生手册（教育管理篇）2025版》PDF。")
    st.divider()

    if init_error:
        st.error("系统初始化失败")
        st.caption(init_error)
    else:
        if init_info.get("status") == "rebuilt":
            st.success("已重建学生手册向量库")
        else:
            st.success("已加载现有学生手册向量库")

        st.caption(init_info.get("message", ""))
        st.caption(f"页数：{init_info.get('page_count')} | 切片数：{init_info.get('chunk_count')}")
        st.caption(f"最近建库时间：{init_info.get('updated_at')}")

    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "你好，我现在只基于《重邮学生手册（教育管理篇）2025版》回答问题。你可以问我奖学金、学籍、处分、申诉等内容。",
        }
    ]


st.title("📘 重邮学生手册智能问答")

if init_error:
    st.error("系统初始化失败，当前无法提供问答服务。")
    st.info("请检查 PDF 文件路径、PyMuPDF 依赖和 `.env` 中的模型密钥配置。")
    st.stop()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "context" in msg:
            if msg.get("pages"):
                st.caption(f"参考页码：{', '.join(map(str, msg['pages']))}")
            with st.expander("查看模型参考的原始资料"):
                st.info(msg["context"] or "这次没有检索到可展示的参考片段。")


if prompt := st.chat_input("请输入你的问题，例如：国家奖学金奖励标准是多少？"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在学生手册中检索并生成回答..."):
                answer, retrieved_context, pages = asyncio.run(ask_question_async(prompt, retriever))

            st.markdown(answer)
            if pages:
                st.caption(f"参考页码：{', '.join(map(str, pages))}")

            with st.expander("查看模型参考的原始资料"):
                st.info(retrieved_context or "这次没有检索到可展示的参考片段。")

        except ValueError as exc:
            answer = f"输入有误：{exc}"
            retrieved_context = ""
            pages = []
            st.warning(answer)
        except RuntimeError as exc:
            answer = f"抱歉，模型调用失败：{exc}"
            retrieved_context = ""
            pages = []
            st.error(answer)
        except Exception as exc:
            answer = f"抱歉，系统出现未预期错误：{exc}"
            retrieved_context = ""
            pages = []
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "context": retrieved_context,
            "pages": pages,
        }
    )
