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

from settings import (
    MODEL_NAME,
    RETRIEVAL_TOP_K,
    CANDIDATE_TOP_K,
    CONTEXT_MAX_CHARS,
    HYBRID_FUSION_TOP_K,
    REWRITE_MODE,
    REWRITE_EXPANSION_WEIGHT,
    NEIGHBOR_PAGE_RADIUS,
    NEIGHBOR_SEED_TOP_N,
)
from vector_store import create_or_load_retriever
from services.llm import get_llm_client
from services.hybrid import reciprocal_rank_fusion
from services.query_normalize import (
    extract_query_keywords as _extract_query_keywords,
    lexical_expand_query as _lexical_expand_query,
)


load_dotenv()

# 全局 BM25 索引，按需初始化
_bm25_index = None

# 当前检索策略，可通过 set_strategy() 切换
_current_strategy = "hybrid"
_current_rewrite_mode = REWRITE_MODE


def set_strategy(strategy: str) -> None:
    global _current_strategy
    _current_strategy = strategy


def get_strategy() -> str:
    return _current_strategy


def set_rewrite_mode(mode: str) -> None:
    global _current_rewrite_mode
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"unsupported rewrite mode: {mode}")
    _current_rewrite_mode = mode


def get_rewrite_mode() -> str:
    return _current_rewrite_mode


def init_rag_system():
    return create_or_load_retriever()


def reset_bm25_index() -> None:
    """Drop in-memory and on-disk BM25 cache (call after vector rebuild)."""
    global _bm25_index
    from pathlib import Path

    _bm25_index = None
    cache_path = Path("bm25_index_cache.json")
    if cache_path.exists():
        cache_path.unlink()


def init_bm25_index():
    """从向量库中加载所有文档构建 BM25 索引。"""
    global _bm25_index
    if _bm25_index is not None and _bm25_index.is_built:
        return _bm25_index

    from pathlib import Path
    from services.hybrid import BM25Index
    from vector_store import load_index_meta

    cache_path = Path("bm25_index_cache.json")
    index_meta = load_index_meta()
    expect_parent_child = str(index_meta.get("splitter_version") or "").startswith("parent_child")
    if cache_path.exists():
        _bm25_index = BM25Index.load(cache_path)
        docs = getattr(_bm25_index, "_documents", []) or []
        has_policy_metadata = all(doc.metadata.get("document_id") for doc in docs)
        has_child_metadata = any(doc.metadata.get("chunk_type") == "child" for doc in docs)
        if _bm25_index.is_built and has_policy_metadata and (has_child_metadata or not expect_parent_child):
            return _bm25_index
        # Stale cache from previous splitter version.
        _bm25_index = None

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


def _finalize_retrieved_docs(docs: list) -> list:
    """Expand child hits to parent context when parent/child index is active."""
    if not docs:
        return []
    if not any((getattr(doc, "metadata", {}) or {}).get("chunk_type") == "child" for doc in docs):
        return docs
    from services.parent_child_index import expand_children_to_parents

    return expand_children_to_parents(docs)


def format_context(docs, *, max_chars: int | None = None) -> tuple[str, list[int]]:
    budget = CONTEXT_MAX_CHARS if max_chars is None else max_chars
    context_parts = []
    source_pages = []
    used = 0
    for doc in docs:
        page = doc.metadata.get("page")
        document_name = doc.metadata.get("document_name") or "学生手册"
        page_text = doc.page_content.strip()
        if not page_text:
            continue
        part = (
            f"【{document_name}｜第 {page} 页】\n{page_text}"
            if page is not None
            else f"【{document_name}】\n{page_text}"
        )
        # Always keep the first (strongest) doc; afterwards respect the budget.
        if context_parts and used + len(part) > budget:
            break
        used += len(part)
        if page is not None:
            source_pages.append(page)
        context_parts.append(part)
    return "\n\n".join(context_parts), sorted(set(source_pages))


def extract_query_keywords(question: str) -> list[str]:
    """Public keyword helper used by retrieval probes and rerank."""
    return _extract_query_keywords(question)


def lexical_expand_query(question: str) -> str:
    """Expand colloquial query with formal campus policy terms."""
    return _lexical_expand_query(question)


def rerank_documents(question: str, docs, *, top_k: int | None = None) -> list:
    """Rerank candidates by original-question keywords and content overlap."""
    limit = RETRIEVAL_TOP_K if top_k is None else top_k
    if not docs:
        return []
    keywords = extract_query_keywords(question)
    question_text = (question or "").strip()

    scored_docs = []
    for index, doc in enumerate(docs):
        text = doc.page_content or ""
        title = str((doc.metadata or {}).get("document_name") or "")
        article_title = str((doc.metadata or {}).get("article_title") or "")
        haystack = f"{title}\n{article_title}\n{text}"
        lexical_score = 0
        for kw in keywords:
            if kw and kw in haystack:
                # Longer policy terms dominate short generic ones.
                lexical_score += max(6, len(kw))
                # Light bonus when the keyword appears more than once.
                lexical_score += min(3, haystack.count(kw) - 1)
        # Prefer pages that actually state sanctions / numeric rules when the
        # user asks about 惩罚/处分/扣分/后果 — helps neighbor-expanded pages surface.
        if any(token in question_text for token in ("惩罚", "处分", "扣分", "处理", "后果", "影响", "严重")):
            for token in ("扣分", "处分", "处理", "晚归", "夜不归宿"):
                if token in haystack:
                    lexical_score += 4
        scored_docs.append((lexical_score * 100 - index, doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    if not keywords or all(score <= 0 for score, _ in scored_docs):
        # Keep hybrid order when overlap signal is empty/flat.
        return list(docs[:limit])
    return [doc for _, doc in scored_docs[:limit]]


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
    docs = _finalize_retrieved_docs(retriever.invoke(question))
    return docs[:RETRIEVAL_TOP_K]


def _retrieve_rerank(question: str, retriever) -> list:
    """策略 rerank：纯向量检索 + 关键词重排序。"""
    candidate_docs = _finalize_retrieved_docs(retriever.invoke(question))
    return rerank_documents(question, candidate_docs)


def _retrieve_rewrite(question: str, retriever) -> list:
    """策略 rewrite：查询改写 + 向量检索 + 关键词重排序。"""
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

    return rerank_documents(question, _finalize_retrieved_docs(all_docs))


def _hybrid_search_once(question: str, retriever, bm25) -> list:
    """Single-query vector+BM25 hybrid search over an expanded candidate pool."""
    from services.hybrid import hybrid_search

    # Lexical expand helps BM25/vector hit formal policy wording while keeping
    # the original utterance. Final ranking still uses the original question.
    query = lexical_expand_query(question)
    child_hits = hybrid_search(
        query=query,
        vector_retriever=retriever,
        bm25_index=bm25,
        top_k_vector=CANDIDATE_TOP_K,
        top_k_bm25=CANDIDATE_TOP_K,
        final_top_k=max(HYBRID_FUSION_TOP_K, RETRIEVAL_TOP_K),
    )
    return _finalize_retrieved_docs(child_hits)


def _probe_query_scores(question: str, retriever, docs: list) -> tuple:
    """Probe vector/BM25/keyword scores for rewrite gating."""
    from services.retrieval import probe_retrieval_scores

    return probe_retrieval_scores(question, retriever, docs)


def _with_neighbor_pages(docs: list) -> list:
    """Expand top hits with same-document neighbor pages before final rerank."""
    if not docs or NEIGHBOR_PAGE_RADIUS <= 0:
        return docs
    from services.parent_child_index import expand_neighbor_pages

    return expand_neighbor_pages(
        docs,
        radius=NEIGHBOR_PAGE_RADIUS,
        max_seed_docs=NEIGHBOR_SEED_TOP_N,
    )


def _retrieve_hybrid(question: str, retriever, rewrite_mode: str | None = None) -> list:
    """策略 hybrid：扩候选 + 邻页补全 + 原问题精排；改写默认 auto。"""
    mode = rewrite_mode or get_rewrite_mode()
    bm25 = init_bm25_index()
    base_docs = _hybrid_search_once(question, retriever, bm25)

    if mode == "off":
        return rerank_documents(question, _with_neighbor_pages(base_docs))

    if mode == "auto":
        # Probe on original question + current candidate context.
        vector_score, bm25_score, keyword_coverage = _probe_query_scores(
            question, retriever, base_docs[:RETRIEVAL_TOP_K]
        )
        from services.retrieval import should_rewrite_after_base_retrieval

        if not should_rewrite_after_base_retrieval(
            vector_score=vector_score,
            bm25_score=bm25_score,
            keyword_coverage=keyword_coverage,
        ):
            return rerank_documents(question, _with_neighbor_pages(base_docs))

    # mode == on, or auto with weak base hit: one gated rewrite pass.
    sub_queries = _rewrite_or_original(question)
    expansions = [q for q in sub_queries if q != question][:1]
    if not expansions:
        return rerank_documents(question, _with_neighbor_pages(base_docs))

    result_sets = [base_docs]
    weights = [1.0]
    for q in expansions:
        result_sets.append(_hybrid_search_once(q, retriever, bm25))
        weights.append(REWRITE_EXPANSION_WEIGHT)

    fused = reciprocal_rank_fusion(
        result_sets,
        k=60,
        top_k=max(HYBRID_FUSION_TOP_K, RETRIEVAL_TOP_K * 2),
        weights=weights,
    )
    # Neighbor pages first, then original-question lexical rerank.
    return rerank_documents(question, _with_neighbor_pages(fused))


def _rewrite_or_original(question: str) -> list[str]:
    """Keep retrieval available when the optional rewrite request is unavailable."""
    from services.rewriter import rewrite_query

    try:
        return rewrite_query(get_llm_client(), question)
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
        client = get_llm_client()
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
