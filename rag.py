"""
RAG 主流程模块。

支持多种检索策略切换：
- "baseline":    纯向量检索 + 原始排序
- "rerank":      纯向量检索 + 关键词重排序（原有方案）
- "rewrite":     查询改写 + 向量检索 + 关键词重排序
- "hybrid":      查询改写 + 向量+BM25混合检索 + RRF融合排序
"""

from __future__ import annotations

import asyncio
import logging
import re

from dotenv import load_dotenv

from settings import MODEL_NAME, RETRIEVAL_TOP_K, CANDIDATE_TOP_K
from vector_store import create_or_load_retriever
from services.llm import get_glm_client


load_dotenv()

# 全局 BM25 索引，按需初始化
_bm25_index = None

# 当前检索策略，可通过 set_strategy() 切换
_current_strategy = "hybrid"


def set_strategy(strategy: str) -> None:
    global _current_strategy
    _current_strategy = strategy


def get_strategy() -> str:
    return _current_strategy


def init_rag_system():
    return create_or_load_retriever()


def init_bm25_index():
    """从向量库中加载所有文档构建 BM25 索引。"""
    global _bm25_index
    if _bm25_index is not None and _bm25_index.is_built:
        return _bm25_index

    from pathlib import Path
    from services.hybrid import BM25Index

    cache_path = Path("bm25_index_cache.json")
    if cache_path.exists():
        _bm25_index = BM25Index.load(cache_path)
        has_policy_metadata = all(
            doc.metadata.get("document_id")
            for doc in getattr(_bm25_index, "_documents", [])
        )
        if _bm25_index.is_built and has_policy_metadata:
            return _bm25_index

    # 从 Chroma 读取所有文档构建
    from langchain_community.vectorstores import Chroma
    from vector_store import get_embeddings, PERSIST_DIRECTORY, COLLECTION_NAME

    vector_store = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
    all_docs = vector_store.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    _bm25_index = BM25Index(documents)
    _bm25_index.save(cache_path)
    return _bm25_index


def format_context(docs) -> tuple[str, list[int]]:
    context_parts = []
    source_pages = []
    for doc in docs:
        page = doc.metadata.get("page")
        document_name = doc.metadata.get("document_name") or "学生手册"
        page_text = doc.page_content.strip()
        if not page_text:
            continue
        if page is not None:
            source_pages.append(page)
            context_parts.append(f"【{document_name}｜第 {page} 页】\n{page_text}")
        else:
            context_parts.append(f"【{document_name}】\n{page_text}")
    return "\n\n".join(context_parts), sorted(set(source_pages))


def extract_query_keywords(question: str) -> list[str]:
    candidate_terms = [
        "学业奖学金", "国家奖学金", "国家励志奖学金", "科创文体奖学金",
        "郭长波奖学金", "社会奖学金", "一等奖", "二等奖", "三等奖",
        "综合素质测评", "综测", "智育", "互斥", "兼得", "同时获得",
        "家庭经济困难", "学科竞赛", "科研成果", "申请条件",
        "退学", "休学", "复学", "转学", "处分", "申诉", "补考", "重修",
        "学籍", "毕业", "学位", "请假", "考勤", "旷课", "奖学金",
    ]
    keywords = [term for term in candidate_terms if term in question]
    if keywords:
        return keywords
    return re.findall(r"[一-鿿]{2,}", question)[:5]


def rerank_documents(question: str, docs) -> list:
    keywords = extract_query_keywords(question)
    scored_docs = []
    for index, doc in enumerate(docs):
        text = doc.page_content
        lexical_score = sum(max(5, len(kw)) for kw in keywords if kw in text)
        scored_docs.append((lexical_score * 100 - index, doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    if all(score <= 0 for score, _ in scored_docs):
        return list(docs[:RETRIEVAL_TOP_K])
    return [doc for _, doc in scored_docs[:RETRIEVAL_TOP_K]]


def build_prompt(question: str, context: str) -> str:
    return f"""你是一名重庆邮电大学奖助政策与学生管理问答助手，请严格根据以下资料回答问题。
要求：
1. 只能根据提供的政策资料作答，不要使用资料外的常识自行补充。
2. 如果资料中没有明确答案，请直接回答"无法确定"。
3. 涉及奖学金条件时，优先按“结论、条件拆解、互斥/限制、依据文件”组织回答。
4. 如果答案依赖综合素质测评、智育成绩、家庭经济困难认定或社会奖学金专项规则，请明确指出需要参考对应文件。

【政策资料】
{context}

【用户问题】
{question}

【回答】"""


def _retrieve_baseline(question: str, retriever) -> list:
    """策略 baseline：纯向量检索，取 Top-K，不做任何重排序。"""
    docs = retriever.invoke(question)
    return docs[:RETRIEVAL_TOP_K]


def _retrieve_rerank(question: str, retriever) -> list:
    """策略 rerank：纯向量检索 + 关键词重排序。"""
    candidate_docs = retriever.invoke(question)
    return rerank_documents(question, candidate_docs)


def _retrieve_rewrite(question: str, retriever) -> list:
    """策略 rewrite：查询改写 + 向量检索 + 关键词重排序。"""
    from services.rewriter import rewrite_query
    sub_queries = _rewrite_or_original(question)

    # 多个子查询分别检索，合并去重
    seen = set()
    all_docs = []
    for q in sub_queries:
        for doc in retriever.invoke(q):
            key = hash(doc.page_content)
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    return rerank_documents(question, all_docs)


def _retrieve_hybrid(question: str, retriever) -> list:
    """策略 hybrid：查询改写 + 向量+BM25混合检索 + RRF融合排序。"""
    from services.rewriter import rewrite_query
    from services.hybrid import hybrid_search

    sub_queries = _rewrite_or_original(question)

    bm25 = init_bm25_index()

    # 每个子查询做混合检索，结果合并
    seen = set()
    all_result_sets = []
    for q in sub_queries:
        docs = hybrid_search(
            query=q,
            vector_retriever=retriever,
            bm25_index=bm25,
            top_k_vector=CANDIDATE_TOP_K,
            top_k_bm25=CANDIDATE_TOP_K,
            final_top_k=RETRIEVAL_TOP_K,
        )
        unique = []
        for doc in docs:
            key = hash(doc.page_content)
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        all_result_sets.append(unique)

    # 把所有子查询的结果再做一次 RRF 融合
    if len(all_result_sets) <= 1:
        return all_result_sets[0] if all_result_sets else []

    from services.hybrid import reciprocal_rank_fusion
    return reciprocal_rank_fusion(all_result_sets, k=60, top_k=RETRIEVAL_TOP_K)


def _rewrite_or_original(question: str) -> list[str]:
    """Keep retrieval available when the optional rewrite request is unavailable."""
    from services.rewriter import rewrite_query

    try:
        return rewrite_query(get_glm_client(), question)
    except Exception:
        logging.warning("Query rewrite unavailable; using original query", exc_info=True)
        return [question]


def ask_question(question: str, retriever) -> tuple[str, str, list[int]]:
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    if retriever is None:
        raise ValueError("检索器尚未初始化")

    strategy = get_strategy()
    if strategy == "baseline":
        docs = _retrieve_baseline(question, retriever)
    elif strategy == "rewrite":
        docs = _retrieve_rewrite(question, retriever)
    elif strategy == "hybrid":
        docs = _retrieve_hybrid(question, retriever)
    else:
        docs = _retrieve_rerank(question, retriever)

    context, source_pages = format_context(docs)
    prompt = build_prompt(question, context)

    try:
        client = get_glm_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = response.choices[0].message.content
        return answer, context, source_pages
    except Exception as exc:
        raise RuntimeError(f"调用大模型失败：{exc}") from exc


async def ask_question_async(question: str, retriever):
    return await asyncio.to_thread(ask_question, question, retriever)


async def get_rag_context_async(question: str, retriever) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    if retriever is None:
        raise ValueError("检索器尚未初始化")

    strategy = get_strategy()
    if strategy == "baseline":
        docs_fn = lambda: _retrieve_baseline(question, retriever)
    elif strategy == "rewrite":
        docs_fn = lambda: _retrieve_rewrite(question, retriever)
    elif strategy == "hybrid":
        docs_fn = lambda: _retrieve_hybrid(question, retriever)
    else:
        docs_fn = lambda: _retrieve_rerank(question, retriever)

    def _build() -> str:
        docs = docs_fn()
        context, _ = format_context(docs)
        return context

    return await asyncio.to_thread(_build)


async def init_rag_system_async():
    return await asyncio.to_thread(init_rag_system)
