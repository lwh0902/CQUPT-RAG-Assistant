from rag import get_strategy


def test_runtime_defaults_to_rewrite_enabled_hybrid_retrieval() -> None:
    assert get_strategy() == "hybrid"
