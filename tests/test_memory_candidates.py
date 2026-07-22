"""Tests for memory candidate parse + three-tier write gate."""

from __future__ import annotations

from services.memory_candidates import (
    MemoryCandidatePayload,
    collect_candidates,
    extract_regex_candidates,
    gate_candidate,
    looks_like_stable_info,
    parse_memory_candidates,
)


def test_parser_keeps_at_most_two_safe_candidates() -> None:
    candidates = parse_memory_candidates(
        '{"candidates":[{"memory_type":"preference","memory_key":"answer_style","memory_value":"简洁","reason":"用户明确要求","confidence":0.95,"explicit":true}]}'
    )
    assert len(candidates) == 1
    assert candidates[0].memory_value == "简洁"


def test_parser_rejects_sensitive_values() -> None:
    assert (
        parse_memory_candidates(
            '{"candidates":[{"memory_type":"profile","memory_key":"phone","memory_value":"18128161378","reason":"用户提供"}]}'
        )
        == []
    )


def test_parser_rejects_non_whitelist_keys() -> None:
    assert (
        parse_memory_candidates(
            '{"candidates":[{"memory_type":"profile","memory_key":"hobby","memory_value":"篮球","reason":"x","confidence":0.99,"explicit":true}]}'
        )
        == []
    )


def test_gate_auto_for_high_confidence_explicit() -> None:
    payload = MemoryCandidatePayload(
        memory_type="preference",
        memory_key="answer_style",
        memory_value="简洁",
        confidence=0.95,
        explicit=True,
    )
    gated = gate_candidate(payload)
    assert gated.decision == "auto"


def test_gate_pending_for_mid_confidence() -> None:
    payload = MemoryCandidatePayload(
        memory_type="profile",
        memory_key="major",
        memory_value="软件工程",
        confidence=0.8,
        explicit=True,
    )
    gated = gate_candidate(payload)
    assert gated.decision == "pending"


def test_gate_reject_low_confidence() -> None:
    payload = MemoryCandidatePayload(
        memory_type="profile",
        memory_key="major",
        memory_value="软件工程",
        confidence=0.4,
        explicit=False,
    )
    gated = gate_candidate(payload)
    assert gated.decision == "reject"
    assert gated.reject_reason == "low_confidence"


def test_regex_extracts_style_and_major() -> None:
    items = extract_regex_candidates("我是计算机专业，以后请简洁回答。")
    keys = {(i.memory_type, i.memory_key) for i in items}
    assert ("profile", "major") in keys
    assert ("preference", "answer_style") in keys


def test_emotion_not_stable_info() -> None:
    assert looks_like_stable_info("今天心情一般，好烦") is False
    assert looks_like_stable_info("我是软件工程专业") is True


def test_collect_candidates_regex_without_llm() -> None:
    items = collect_candidates("请以后详细回答", use_llm=False)
    assert len(items) == 1
    assert items[0].memory_key == "answer_style"
    assert items[0].memory_value == "详细"
