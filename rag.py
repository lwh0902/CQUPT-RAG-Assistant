"""
RAG 主流程模块。

这个文件只负责：
1. 获取或初始化 retriever
2. 检索与组装上下文
3. 调用大模型生成回答
4. 提供异步包装，避免上层直接阻塞
"""

import asyncio
import re

from dotenv import load_dotenv
from zhipuai import ZhipuAI

from settings import MODEL_NAME, RETRIEVAL_TOP_K
from vector_store import create_or_load_retriever


load_dotenv()


def get_glm_client() -> ZhipuAI:
    """
    获取智谱客户端。
    """
    from os import getenv

    api_key = getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("未找到 ZHIPU_API_KEY，请检查 .env 文件是否配置正确。")
    return ZhipuAI(api_key=api_key)


def init_rag_system():
    """
    初始化 RAG 系统。

    这里会自动判断：
    - 是否需要根据 PDF 重建向量库
    - 是否可以直接加载已有索引
    """
    return create_or_load_retriever()


def format_context(docs) -> tuple[str, list[int]]:
    """
    把检索结果拼成 Prompt 使用的上下文，同时提取来源页码。
    """
    context_parts = []
    source_pages = []

    for doc in docs:
        page = doc.metadata.get("page")
        page_text = doc.page_content.strip()
        if not page_text:
            continue

        if page is not None:
            source_pages.append(page)
            context_parts.append(f"【第 {page} 页】\n{page_text}")
        else:
            context_parts.append(page_text)

    unique_pages = sorted(set(source_pages))
    context = "\n\n".join(context_parts)
    return context, unique_pages


def extract_query_keywords(question: str) -> list[str]:
    """
    从问题里提取对制度问答更有帮助的关键词。

    这里优先保留较长的专有短语，避免“一等奖”这类泛词权重过高。
    """
    candidate_terms = [
        "学业奖学金",
        "国家奖学金",
        "国家励志奖学金",
        "科创文体奖学金",
        "郭长波奖学金",
        "社会奖学金",
        "一等奖",
        "二等奖",
        "三等奖",
        "退学",
        "休学",
        "复学",
        "转学",
        "处分",
        "申诉",
        "补考",
        "重修",
        "学籍",
        "毕业",
        "学位",
    ]

    keywords = [term for term in candidate_terms if term in question]
    if keywords:
        return keywords

    # 如果没命中预设短语，再退化到较长中文连续片段。
    fallback_terms = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    return fallback_terms[:5]


def rerank_documents(question: str, docs) -> list:
    """
    对向量检索候选结果做一次轻量二次排序。

    思路：
    - 先保留向量检索召回能力
    - 再用问题中的关键词给文档加词面匹配分
    - 最后只保留最相关的前几个 chunk 进入 Prompt
    """
    keywords = extract_query_keywords(question)
    scored_docs = []

    for index, doc in enumerate(docs):
        text = doc.page_content
        lexical_score = 0

        for keyword in keywords:
            if keyword in text:
                # 长短语命中比泛词更重要。
                lexical_score += max(5, len(keyword))

        # 用 index 做一个很小的稳定项，尽量保留原始召回顺序。
        combined_score = lexical_score * 100 - index
        scored_docs.append((combined_score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:RETRIEVAL_TOP_K]]

    # 如果所有候选都没有任何词面命中，就退回原始顺序的前几个。
    if all(score <= 0 for score, _ in scored_docs):
        return list(docs[:RETRIEVAL_TOP_K])

    return top_docs


def build_prompt(question: str, context: str) -> str:
    """
    构造给大模型的最终 Prompt。
    """
    return f"""你是一名重庆邮电大学学生手册问答助手，请严格根据以下资料回答问题。
要求：
1. 只能根据提供的学生手册资料作答，不要使用资料外的常识自行补充。
2. 如果资料中没有明确答案，请直接回答“无法确定”。
3. 回答要简洁清楚，优先提炼结论。

【学生手册资料】
{context}

【用户问题】
{question}

【回答】"""


def ask_question(question: str, retriever):
    """
    执行一次完整问答。

    返回：
    - answer：最终回答
    - context：检索到的上下文
    - source_pages：命中的页码列表
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空。")

    if retriever is None:
        raise ValueError("检索器尚未初始化，无法执行问答。")

    candidate_docs = retriever.invoke(question)
    docs = rerank_documents(question, candidate_docs)
    context, source_pages = format_context(docs)
    prompt = build_prompt(question, context)

    try:
        client = get_glm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content
        return answer, context, source_pages
    except Exception as exc:
        raise RuntimeError(f"调用大模型生成回答失败：{exc}") from exc


async def ask_question_async(question: str, retriever):
    """
    异步包装层。

    由于底层 SDK 和检索流程仍是同步的，这里用 to_thread
    把阻塞调用放到线程里，避免上层事件循环被卡住。
    """
    return await asyncio.to_thread(ask_question, question, retriever)


async def init_rag_system_async():
    """
    异步初始化 RAG 系统。
    """
    return await asyncio.to_thread(init_rag_system)


if __name__ == "__main__":
    print("正在初始化基于学生手册 PDF 的 RAG 系统...")
    try:
        retriever, init_info = init_rag_system()
        print(init_info["message"])
        print(f"索引页数：{init_info['page_count']}，切片数：{init_info['chunk_count']}")
        print("系统就绪！输入问题（输入 '退出' 结束）\n")
    except Exception as exc:
        print(f"初始化失败：{exc}")
        raise SystemExit(1)

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ["退出", "q", "quit"]:
            break
        if not question:
            continue

        try:
            answer, context, pages = ask_question(question, retriever)
            print(f"\n回答：\n{answer}")
            print(f"\n命中页码：{pages or '未命中具体页码'}")
            preview = context[:300] + "..." if context else "未检索到相关片段。"
            print(f"\n检索上下文预览：\n{preview}")
            print("\n" + "=" * 50 + "\n")
        except Exception as exc:
            print(f"\n处理问题时出错：{exc}\n")
