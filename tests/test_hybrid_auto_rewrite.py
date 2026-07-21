from langchain_core.documents import Document

import rag


def test_hybrid_auto_skips_rewrite_when_base_retrieval_is_strong(monkeypatch) -> None:
    base_doc = Document(page_content="奖学金条件", metadata={"document_id": "manual", "page": 2})
    calls = {"rewrite": 0, "hybrid_once": 0}

    monkeypatch.setattr(rag, "init_bm25_index", lambda: object())
    monkeypatch.setattr(
        rag,
        "_hybrid_search_once",
        lambda question, retriever, bm25: calls.__setitem__("hybrid_once", calls["hybrid_once"] + 1) or [base_doc],
    )
    monkeypatch.setattr(
        rag,
        "_probe_query_scores",
        lambda question, retriever, docs: (0.66, 9.0, 1.0),
    )

    def boom(*_args, **_kwargs):
        calls["rewrite"] += 1
        raise AssertionError("rewrite should not run")

    monkeypatch.setattr(rag, "_rewrite_or_original", boom)

    docs = rag._retrieve_hybrid("国家奖学金申请条件是什么", object(), rewrite_mode="auto")

    assert docs == [base_doc]
    assert calls["hybrid_once"] == 1
    assert calls["rewrite"] == 0


def test_hybrid_auto_rewrites_once_and_keeps_base_weight(monkeypatch) -> None:
    base_doc = Document(page_content="口语弱命中", metadata={"document_id": "manual", "page": 1})
    rewrite_doc = Document(page_content="制度术语命中", metadata={"document_id": "manual", "page": 11})
    calls = {"queries": []}
    captured = {}

    monkeypatch.setattr(rag, "init_bm25_index", lambda: object())

    def fake_once(question, retriever, bm25):
        calls["queries"].append(question)
        if question == "挂科了咋办？":
            return [base_doc]
        return [rewrite_doc]

    monkeypatch.setattr(rag, "_hybrid_search_once", fake_once)
    monkeypatch.setattr(
        rag,
        "_probe_query_scores",
        lambda question, retriever, docs: (0.40, 2.0, 0.0),
    )
    monkeypatch.setattr(
        rag,
        "_rewrite_or_original",
        lambda question: [question, "考核不合格 补考"],
    )

    def fake_rrf(result_sets, k=60, top_k=3, weights=None):
        captured["weights"] = weights
        captured["n_sets"] = len(result_sets)
        return [rewrite_doc, base_doc]

    monkeypatch.setattr(rag, "reciprocal_rank_fusion", fake_rrf)

    docs = rag._retrieve_hybrid("挂科了咋办？", object(), rewrite_mode="auto")

    assert docs[0].metadata["page"] == 11
    assert calls["queries"][0] == "挂科了咋办？"
    assert "考核不合格 补考" in calls["queries"]
    assert captured["n_sets"] == 2
    assert captured["weights"][0] > captured["weights"][1]


def test_hybrid_off_never_rewrites(monkeypatch) -> None:
    base_doc = Document(page_content="base", metadata={"document_id": "manual", "page": 3})
    monkeypatch.setattr(rag, "init_bm25_index", lambda: object())
    monkeypatch.setattr(rag, "_hybrid_search_once", lambda q, r, b: [base_doc])
    monkeypatch.setattr(
        rag,
        "_rewrite_or_original",
        lambda q: (_ for _ in ()).throw(AssertionError("no rewrite")),
    )

    docs = rag._retrieve_hybrid("任意问题", object(), rewrite_mode="off")
    assert docs == [base_doc]
