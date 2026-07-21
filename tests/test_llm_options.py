from services import llm


def test_completion_forwards_user_generation_options(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **payload):
            captured.update(payload)
            return object()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm, "get_llm_client", lambda: FakeClient())

    llm.create_llm_completion(
        [{"role": "user", "content": "你好"}],
        with_tools=False,
        temperature=0.6,
        top_p=0.7,
    )

    assert captured["temperature"] == 0.6
    assert captured["top_p"] == 0.7
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
