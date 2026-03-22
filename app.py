"""
Streamlit Web 应用入口。
"""

import streamlit as st

from rag import load_and_split_document, create_vector_db, ask_question


st.set_page_config(
    page_title="CQUPT 智能知识库",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def init_rag_system():
    """
    初始化 RAG 系统。

    这里使用 Streamlit 的资源缓存：
    - 第一次运行时初始化检索器
    - 后续页面交互直接复用，避免反复构建
    """
    with st.spinner("正在初始化向量数据库，请稍候..."):
        chunks = load_and_split_document("cqupt.txt")
        retriever = create_vector_db(chunks)
        return retriever


try:
    retriever = init_rag_system()
    init_error = None
except Exception as exc:
    retriever = None
    init_error = str(exc)


with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/zh/thumb/3/3a/Chongqing_University_of_Posts_and_Telecommunications_logo.svg/1200px-Chongqing_University_of_Posts_and_Telecommunications_logo.svg.png",
        width=150,
    )
    st.markdown("### 🎓 重邮智能问答系统")
    st.markdown("本系统基于 **GLM-4** 与 **LangChain** 构建，采用 RAG 架构进行校园知识问答。")
    st.divider()

    if init_error:
        st.error("系统初始化失败")
        st.caption(init_error)
    else:
        st.success("向量数据库已就绪")
        st.success("模型接口可用")

    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "你好，同学！我是重邮校园 AI 助手，关于校训、图书馆、食堂、校园服务等问题都可以问我。",
        }
    ]


st.title("🤖 重邮校园知识问答")

if init_error:
    st.error("RAG 系统初始化失败，当前无法提供问答服务。")
    st.info("你可以先检查 `.env`、知识库文件 `cqupt.txt`，以及本地模型/向量库相关依赖是否正常。")
    st.stop()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("查看模型参考的原始资料"):
                st.info(msg["context"] or "这次没有检索到可展示的参考片段。")


if prompt := st.chat_input("请输入你的问题，例如：图书馆周末几点关门？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在校园知识库中检索并生成回答..."):
                answer, retrieved_context = ask_question(prompt, retriever)

            st.markdown(answer)
            with st.expander("查看模型参考的原始资料"):
                st.info(retrieved_context or "这次没有检索到可展示的参考片段。")

        except ValueError as exc:
            answer = f"输入有误：{exc}"
            retrieved_context = ""
            st.warning(answer)
        except RuntimeError as exc:
            answer = f"抱歉，模型调用失败：{exc}"
            retrieved_context = ""
            st.error(answer)
        except Exception as exc:
            answer = f"抱歉，系统出现未预期错误：{exc}"
            retrieved_context = ""
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "context": retrieved_context,
        }
    )
