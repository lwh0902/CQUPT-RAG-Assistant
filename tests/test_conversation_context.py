from services.conversation_context import (
    is_followup_utterance,
    previous_user_utterance,
    resolve_followup_query,
    turns_to_chat_messages,
)


def test_followup_detects_deictic_late_return_continuation() -> None:
    prev = "晚归有什么惩罚"
    assert is_followup_utterance("如果直接不回呢", prev) is True
    assert is_followup_utterance("那夜不归宿呢", prev) is True
    resolved = resolve_followup_query("如果直接不回呢", prev)
    assert "晚归有什么惩罚" in resolved
    assert "如果直接不回呢" in resolved


def test_followup_does_not_rewrite_new_standalone_topic() -> None:
    prev = "晚归有什么惩罚"
    current = "国家奖学金申请条件是什么"
    assert is_followup_utterance(current, prev) is False
    assert resolve_followup_query(current, prev) == current


def test_previous_user_utterance_skips_current() -> None:
    turns = [
        {"role": "user", "content": "晚归有什么惩罚"},
        {"role": "assistant", "content": "晚归扣5分"},
        {"role": "user", "content": "如果直接不回呢"},
    ]
    assert previous_user_utterance(turns, "如果直接不回呢") == "晚归有什么惩罚"


def test_turns_to_chat_messages_excludes_trailing_user() -> None:
    turns = [
        {"role": "user", "content": "晚归有什么惩罚"},
        {"role": "assistant", "content": "扣5分"},
        {"role": "user", "content": "如果直接不回呢"},
    ]
    messages = turns_to_chat_messages(turns, exclude_last_user=True)
    assert messages[-1]["role"] == "assistant"
    assert len(messages) == 2
