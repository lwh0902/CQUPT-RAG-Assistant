from rag import get_strategy, get_rewrite_mode


def test_runtime_defaults_to_hybrid_with_auto_rewrite_gate() -> None:
    assert get_strategy() == "hybrid"
    assert get_rewrite_mode() == "auto"
