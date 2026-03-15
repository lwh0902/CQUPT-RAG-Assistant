"""
工业级 RAG Web 应用 - 重邮智能助手
文件名: app.py
"""
import streamlit as st
import os
from dotenv import load_dotenv

# 直接复用我们之前写好的核心大脑！这就是代码解耦的好处！
from rag import load_and_split_document, create_vector_db, ask_question

# ============ 1. 页面全局配置 ============
st.set_page_config(
    page_title="CQUPT 智能知识库",
    page_icon="🎓",
    layout="centered"
)

# ============ 2. 核心资源缓存 (极其重要) ============
# @st.cache_resource 告诉 Streamlit：这个函数极其耗时，执行一次后请把返回值(retriever)死死存在内存里！
# 无论用户怎么疯狂点击界面，都不要再重新执行这个函数了。
@st.cache_resource
def init_rag_system():
    with st.spinner("⏳ 正在初始化向量数据库，请稍候..."):
        # 假设你的文件叫 cqupt.txt，确保它在同级目录下
        chunks = load_and_split_document("cqupt.txt")
        retriever = create_vector_db(chunks)
        return retriever

# 调用缓存的初始化函数
retriever = init_rag_system()

# ============ 3. 侧边栏设计 (展现专业度) ============
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/zh/thumb/3/3a/Chongqing_University_of_Posts_and_Telecommunications_logo.svg/1200px-Chongqing_University_of_Posts_and_Telecommunications_logo.svg.png", width=150)
    st.markdown("### 🎓 重邮智能问答系统")
    st.markdown("本系统基于 **GLM-4** 与 **LangChain** 强力驱动，采用 RAG 架构构建。")
    st.divider()
    st.success("✅ 向量数据库已连接")
    st.success("✅ 大模型 API 已就绪")
    
    # 一个清空历史记录的实用小按钮
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.rerun()

# ============ 4. 状态管理 (Session State) ============
# 检查当前用户的浏览器会话中是否已经有 "messages" 这个变量，没有就建一个空列表
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，同学！我是重邮专属 AI 助手，关于校训、图书馆、食堂等信息，尽管问我吧！"}
    ]

# ============ 5. 主界面渲染 ============
st.title("🤖 邮子帮·校园百事通")

# 遍历并展示历史聊天记录
for msg in st.session_state.messages:
    # st.chat_message 是 Streamlit 内置的绝佳聊天UI组件
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 这是一个高阶加分项：如果是 AI 的回答，我们可以把“检索到的参考资料”折叠展示出来
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("🔍 查看模型参考的原始文档"):
                st.info(msg["context"])

# ============ 6. 用户输入与交互逻辑 ============
# st.chat_input 会在页面最底部固定一个输入框
if prompt := st.chat_input("请输入你的问题（例如：图书馆几点关门？）"):
    
    # 第一步：把用户输入立刻显示在界面上，并存入历史记录
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 第二步：调用 RAG 大脑生成回答
    with st.chat_message("assistant"):
        # 用一个加载动画安抚用户等待的焦虑
        with st.spinner("正在校园知识库中极速检索..."):
            # 调用你写好的 rag.py 里的函数！
            answer, retrieved_context = ask_question(prompt, retriever)
            
            # 显示回答
            st.markdown(answer)
            # 以折叠框形式展示检索到的证据，增加大模型回答的可信度（解决幻觉痛点）
            with st.expander("🔍 查看模型参考的原始文档"):
                st.info(retrieved_context)
                
    # 第三步：把 AI 的回答也存入历史记录
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "context": retrieved_context # 把参考资料也存起来，方便翻阅历史时查看
    })