"""Quick fact matching: only verified facts, conservative patterns."""

from __future__ import annotations

import json
from pathlib import Path

from services.quick_facts import (
    list_public_facts,
    match_quick_fact,
    render_quick_fact_reply,
)


def _write_facts(tmp_path: Path, facts: list[dict]) -> Path:
    path = tmp_path / "quick_facts.json"
    path.write_text(json.dumps({"version": 1, "facts": facts}, ensure_ascii=False))
    return path


def _fact(**overrides) -> dict:
    base = {
        "id": "jwc_phone",
        "title": "教务处联系电话",
        "sample_question": "教务处电话多少",
        "patterns": [r"教务处.{0,6}(电话|联系方式|怎么联系|联系)", r"(电话|联系方式|联系).{0,6}教务处"],
        "answer": "023-00000000",
        "source_name": "重庆邮电大学教务处",
        "source_url": "https://jwc.cqupt.edu.cn/",
        "updated_at": "2025-09",
        "verified": True,
    }
    base.update(overrides)
    return base


def test_verified_fact_matches_common_phrasings(tmp_path) -> None:
    path = _write_facts(tmp_path, [_fact()])
    assert match_quick_fact("教务处电话多少", path=path) is not None
    assert match_quick_fact("怎么联系教务处啊", path=path) is not None
    assert match_quick_fact("教务处的联系方式", path=path) is not None


def test_unverified_fact_never_matches(tmp_path) -> None:
    path = _write_facts(tmp_path, [_fact(verified=False)])
    assert match_quick_fact("教务处电话多少", path=path) is None
    assert list_public_facts(path=path) == []


def test_unrelated_query_does_not_match(tmp_path) -> None:
    path = _write_facts(tmp_path, [_fact()])
    assert match_quick_fact("晚归有什么惩罚", path=path) is None


def test_long_query_does_not_match(tmp_path) -> None:
    path = _write_facts(tmp_path, [_fact()])
    long_query = "教务处电话" + "很长的问题" * 40
    assert match_quick_fact(long_query, path=path) is None


def test_render_reply_includes_source(tmp_path) -> None:
    path = _write_facts(tmp_path, [_fact()])
    fact = match_quick_fact("教务处电话多少", path=path)
    reply = render_quick_fact_reply(fact)
    assert "教务处联系电话" in reply
    assert "023-00000000" in reply
    assert "重庆邮电大学教务处" in reply
    assert "2025-09" in reply


def test_shipped_facts_verified_gate_works() -> None:
    """Repo data: verified facts match; unverified stubs stay invisible."""
    public = list_public_facts()
    public_ids = {item["id"] for item in public}
    # Unverified stubs must never leak.
    assert "service_hall_hours" not in public_ids
    assert "xg_phone" not in public_ids
    assert match_quick_fact("办事大厅几点上班") is None
    assert match_quick_fact("学工部电话多少") is None
    # Verified facts answer common phrasings.
    assert match_quick_fact("教务处电话多少") is not None
    assert match_quick_fact("怎么联系教务处") is not None
    assert match_quick_fact("什么时候放寒假") is not None
    assert match_quick_fact("图书馆几点开门") is not None
    # Calendar patterns must not swallow policy questions.
    assert match_quick_fact("放寒假前补考怎么安排") is None
    assert match_quick_fact("晚归有什么惩罚") is None
