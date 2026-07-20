from pathlib import Path


def test_hybrid_retrieval_dependencies_are_declared() -> None:
    dependencies = {
        line.strip().lower()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert "jieba" in dependencies
    assert "rank-bm25" in dependencies
