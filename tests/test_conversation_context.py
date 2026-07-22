from __future__ import annotations

from services.conversation_context import (
    is_followup_utterance,
    previous_user_utterance,
    resolve_followup_query,
    resolve_followup_query_llm,
    turns_to_chat_messages,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(content, error)


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


def test_llm_resolve_uses_clean_standalone_query(monkeypatch) -> None:
    import services.llm as llm_mod

    client = _FakeClient("晚归的公寓行为积分扣多少分")
    monkeypatch.setattr(llm_mod, "get_llm_client", lambda: client)

    resolved = resolve_followup_query_llm("扣多少分", "我如果回宿舍比较晚怎么办 就是晚归")

    assert resolved == "晚归的公寓行为积分扣多少分"
    assert "用户追问" not in resolved
    assert client.chat.completions.calls == 1


def test_llm_resolve_falls_back_on_error(monkeypatch) -> None:
    import services.llm as llm_mod

    client = _FakeClient(error=RuntimeError("boom"))
    monkeypatch.setattr(llm_mod, "get_llm_client", lambda: client)

    resolved = resolve_followup_query_llm("扣多少分", "晚归有什么惩罚")

    assert resolved == "晚归有什么惩罚\n用户追问：扣多少分"


def test_llm_resolve_falls_back_on_garbage_output(monkeypatch) -> None:
    import services.llm as llm_mod

    client = _FakeClient("第一行\n第二行")
    monkeypatch.setattr(llm_mod, "get_llm_client", lambda: client)

    resolved = resolve_followup_query_llm("扣多少分", "晚归有什么惩罚")

    assert resolved == "晚归有什么惩罚\n用户追问：扣多少分"


def test_llm_resolve_skips_llm_for_standalone_question(monkeypatch) -> None:
    import services.llm as llm_mod

    client = _FakeClient("不该被调用")
    monkeypatch.setattr(llm_mod, "get_llm_client", lambda: client)

    resolved = resolve_followup_query_llm("国家奖学金申请条件是什么", "晚归有什么惩罚")

    assert resolved == "国家奖学金申请条件是什么"
    assert client.chat.completions.calls == 0


def test_llm_resolve_disabled_uses_rule_path() -> None:
    resolved = resolve_followup_query_llm("扣多少分", "晚归有什么惩罚", enabled=False)
    assert resolved == "晚归有什么惩罚\n用户追问：扣多少分"
