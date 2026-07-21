from services import session_summary


def test_session_summary_returns_structured_user_visible_result(monkeypatch) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": '{"topic":"奖学金申请","confirmed_points":["需准备材料"],"open_questions":[],"next_actions":["提交申请"]}'
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(session_summary, "get_llm_client", lambda: FakeClient())

    result = session_summary.summarize_conversation([
        {"role": "user", "content": "奖学金怎么申请？"},
        {"role": "assistant", "content": "需要准备材料。"},
    ])

    assert result.topic == "奖学金申请"
    assert result.next_actions == ["提交申请"]
