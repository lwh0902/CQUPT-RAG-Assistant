from services.retrieval import should_rewrite_after_base_retrieval


def test_strong_base_retrieval_skips_rewrite() -> None:
    assert (
        should_rewrite_after_base_retrieval(
            vector_score=0.62,
            bm25_score=8.0,
            keyword_coverage=1.0,
        )
        is False
    )


def test_weak_in_domain_retrieval_triggers_rewrite() -> None:
    assert (
        should_rewrite_after_base_retrieval(
            vector_score=0.41,
            bm25_score=3.2,
            keyword_coverage=0.0,
        )
        is True
    )


def test_clear_out_of_scope_skips_rewrite() -> None:
    assert (
        should_rewrite_after_base_retrieval(
            vector_score=0.12,
            bm25_score=5.7,
            keyword_coverage=0.0,
        )
        is False
    )


def test_missing_vector_score_skips_rewrite_to_protect_precise_queries() -> None:
    assert (
        should_rewrite_after_base_retrieval(
            vector_score=None,
            bm25_score=1.0,
            keyword_coverage=0.0,
        )
        is False
    )
