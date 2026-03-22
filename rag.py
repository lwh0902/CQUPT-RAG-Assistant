"""
RAG 核心模块。

这个文件负责整个问答主流程：
1. 读取本地知识库文本
2. 将长文本切成较小片段
3. 首次构建或后续加载 Chroma 向量库
4. 根据用户问题检索相关片段
5. 把检索结果交给大模型生成最终回答
"""

import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


def get_glm_client() -> ZhipuAI:
    """
    获取智谱客户端。

    单独封装成函数后：
    1. 出错时能给出更明确的提示
    2. 不会在模块导入时就因为缺少环境变量而直接崩掉
    """
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("未找到 ZHIPU_API_KEY，请检查 .env 文件是否配置正确。")
    return ZhipuAI(api_key=api_key)


def get_embeddings():
    """
    创建中文 Embedding 模型。

    这里单独抽成函数，是为了让“首次构建”和“后续加载”都能复用同一套配置。
    """
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split_document(file_path="cqupt.txt"):
    """
    读取 txt 知识库，并把它切成多个带少量重叠的文本片段。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到知识库文件：{file_path}，请先创建！")

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # 切片策略说明：
    # chunk_size=100：每个片段大约 100 个字符，便于精确检索
    # chunk_overlap=20：相邻片段保留 20 个字符重叠，避免语义断裂
    # separators：优先按段落、换行、句号等更自然的位置切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    return text_splitter.split_documents(docs)


def _vector_db_exists(persist_dir: str) -> bool:
    """
    判断指定目录下是否已经存在可复用的 Chroma 数据。
    """
    sqlite_file = os.path.join(persist_dir, "chroma.sqlite3")
    return os.path.exists(sqlite_file)


def create_vector_db(chunks, persist_dir="./chroma_db"):
    """
    首次构建或后续加载 Chroma 向量库，并返回 retriever。

    现在的逻辑是：
    - 如果本地已经有持久化向量库，就直接加载
    - 如果没有，再根据 chunks 首次构建
    """
    embeddings = get_embeddings()

    if _vector_db_exists(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3},
    )


def ask_question(question, retriever):
    """
    执行一次完整的 RAG 问答流程。

    返回值：
        (answer, context)
        answer：模型生成的最终答案
        context：检索到的原始文本，方便界面展示或调试
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空。")

    if retriever is None:
        raise ValueError("检索器尚未初始化，无法执行问答。")

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""你是一名重庆邮电大学助手，请严格根据以下资料回答问题。
要求：
1. 仅提取与用户问题直接相关的信息，不要回答资料中的无关内容。
2. 如果资料中没有相关信息，请直接回答“无法确定”，绝不能编造答案。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

    try:
        client = get_glm_client()
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content
        return answer, context
    except Exception as exc:
        raise RuntimeError(f"调用大模型生成回答失败：{exc}") from exc


if __name__ == "__main__":
    print("正在初始化 RAG 系统（如需首次构建向量库，请耐心等待）...")
    try:
        chunks = load_and_split_document()
        retriever = create_vector_db(chunks)
        print("系统就绪！输入问题（输入 '退出' 结束）\n")
    except Exception as exc:
        print(f"初始化失败：{exc}")
        raise SystemExit(1)

    while True:
        q = input("你的问题：").strip()
        if q.lower() in ["退出", "q", "quit"]:
            break
        if not q:
            continue

        try:
            answer, retrieved = ask_question(q, retriever)
            print(f"\n回答：\n{answer}")
            preview = retrieved[:200] + "..." if retrieved else "未检索到相关片段，模型将根据 Prompt 规则拒绝回答。"
            print(f"\n检索到的原文片段：\n{preview}")
            print("\n" + "=" * 50 + "\n")
        except Exception as exc:
            print(f"\n处理问题时出错：{exc}\n")
