from services.memory_candidates import parse_memory_candidates


def test_parser_keeps_at_most_two_safe_candidates() -> None:
    candidates = parse_memory_candidates(
        '{"candidates":[{"memory_type":"preference","memory_key":"answer_style","memory_value":"简洁","reason":"用户明确要求"}]}'
    )
    assert len(candidates) == 1
    assert candidates[0].memory_value == "简洁"


def test_parser_rejects_sensitive_values() -> None:
    assert parse_memory_candidates(
        '{"candidates":[{"memory_type":"profile","memory_key":"phone","memory_value":"18128161378","reason":"用户提供"}]}'
    ) == []
