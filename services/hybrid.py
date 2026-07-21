"""混合检索模块：向量检索 + BM25 关键词检索 + RRF 融合排序。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# 加载校园相关自定义词典，提升专有词汇分词准确率
jieba.initialize()


def _tokenize_chinese(text: str) -> list[str]:
    """jieba 精确模式分词，过滤停用词和单字。"""
    tokens = list(jieba.cut(text))
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


class BM25Index:
    """基于 rank_bm25 的 BM25 索引，支持持久化。"""

    def __init__(self, documents: list[Document] | None = None):
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if documents:
            self.build(documents)

    def build(self, documents: list[Document]) -> None:
        self._documents = documents
        self._tokenized_corpus = [_tokenize_chinese(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        if self._bm25 is None:
            return []
        tokenized_query = _tokenize_chinese(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._documents[i], float(score)) for i, score in ranked[:top_k] if score > 0]

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    def save(self, path: Path) -> None:
        """持久化索引数据。"""
        data = {
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self._documents
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """从文件加载索引。"""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in data["documents"]
        ]
        index = cls()
        if documents:
            index.build(documents)
        return index


def reciprocal_rank_fusion(
    result_sets: list[list[Document]],
    k: int = 60,
    top_k: int = 3,
    weights: list[float] | None = None,
) -> list[Document]:
    """
    RRF 融合：多路检索结果按排名倒数加权合并。

    score(doc) = Σ weight_i / (k + rank_i)
    """
    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}
    if weights is None:
        weights = [1.0] * len(result_sets)
    if len(weights) != len(result_sets):
        raise ValueError("weights length must match result_sets length")

    for results, weight in zip(result_sets, weights):
        for rank, doc in enumerate(results):
            # 用 content hash 作为稳定 key（id() 可能重复）
            stable_key = hash(doc.page_content)
            doc_map[stable_key] = doc
            rrf_scores[stable_key] = rrf_scores.get(stable_key, 0.0) + float(weight) / (k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:top_k]]


def hybrid_search(
    query: str,
    vector_retriever,
    bm25_index: BM25Index,
    top_k_vector: int = 8,
    top_k_bm25: int = 8,
    final_top_k: int = 3,
) -> list[Document]:
    """执行向量 + BM25 双路检索，然后用 RRF 融合。"""
    # 路径 1: 向量检索
    vector_docs = vector_retriever.invoke(query)[:top_k_vector]

    # 路径 2: BM25 检索
    bm25_results = bm25_index.search(query, top_k=top_k_bm25)
    bm25_docs = [doc for doc, _ in bm25_results]

    # RRF 融合
    return reciprocal_rank_fusion(
        [vector_docs, bm25_docs],
        k=60,
        top_k=final_top_k,
    )
