from langchain_core.documents import Document

import rag
from settings import RETRIEVAL_TOP_K


def test_hybrid_search_once_requests_expanded_candidate_pool(monkeypatch) -> None:
    captured = {}

    def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return [
            Document(page_content=f"d{i}", metadata={"page": i, "chunk_type": "legacy"})
            for i in range(kwargs["final_top_k"])
        ]

    monkeypatch.setattr("services.hybrid.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(rag, "_finalize_retrieved_docs", lambda docs: docs)
    docs = rag._hybrid_search_once("国家奖学金申请条件", object(), object())

    assert captured["top_k_vector"] >= 12
    assert captured["top_k_bm25"] >= 12
    assert captured["final_top_k"] >= 12
    assert len(docs) == captured["final_top_k"]


def test_hybrid_off_reranks_original_question_over_expanded_candidates(monkeypatch) -> None:
    """Even without rewrite, hybrid should rerank a wider candidate list by original query."""
    docs = [
        Document(page_content="五星文明楼评选条件……", metadata={"page": 77}),
        Document(page_content="卫生寝室评选条件：每学期安全卫生检查平均分……", metadata={"page": 76}),
        Document(page_content="无关宿舍管理通知", metadata={"page": 144}),
    ]
    # Return more than RETRIEVAL_TOP_K candidates from one-shot hybrid.
    monkeypatch.setattr(rag, "init_bm25_index", lambda: object())
    monkeypatch.setattr(rag, "_hybrid_search_once", lambda q, r, b: docs)
    monkeypatch.setattr(
        rag,
        "_probe_query_scores",
        lambda q, r, d: (0.7, 10.0, 1.0),  # strong enough to skip rewrite
    )

    result = rag._retrieve_hybrid("寝室想评卫生寝室，检查分至少得多少？", object(), rewrite_mode="off")

    assert len(result) == RETRIEVAL_TOP_K
    assert result[0].metadata["page"] == 76


def test_retrieve_hybrid_runs_neighbor_expand_before_final_rerank(monkeypatch) -> None:
    docs = [
        Document(page_content="第十一条 考核", metadata={"document_id": "manual", "page": 11}),
    ]
    monkeypatch.setattr(rag, "init_bm25_index", lambda: object())
    monkeypatch.setattr(rag, "_hybrid_search_once", lambda q, r, b: docs)
    monkeypatch.setattr(rag, "NEIGHBOR_PAGE_RADIUS", 1)
    monkeypatch.setattr(rag, "NEIGHBOR_SEED_TOP_N", 3)

    seen = {}

    def fake_neighbors(items):
        seen["called"] = True
        return items + [
            Document(
                page_content="第十二条 转学",
                metadata={"document_id": "manual", "page": 12, "neighbor_expanded": True},
            )
        ]

    monkeypatch.setattr(rag, "_with_neighbor_pages", fake_neighbors)
    result = rag._retrieve_hybrid("转学", object(), rewrite_mode="off")
    assert seen.get("called") is True
    assert any(doc.metadata.get("page") == 12 for doc in result) or len(result) >= 1
