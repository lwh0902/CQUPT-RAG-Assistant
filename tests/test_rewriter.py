from services.rewriter import REWRITE_PROMPT
from services.rewriter import rewrite_query


def test_rewrite_prompt_includes_colloquial_to_policy_term_examples() -> None:
    prompt = REWRITE_PROMPT.format(question="挂科了咋办？")

    assert "挂科 -> 考核不合格、补考、重修" in prompt
    assert "缓交学费 -> 暂缓注册、助学贷款、资助" in prompt
    assert "第一行必须原样保留用户问题" in prompt


def test_rewrite_query_keeps_original_and_at_most_two_expansions() -> None:
    class FakeCompletions:
        def create(self, **_):
            message = type("Message", (), {"content": "挂科了咋办？\n考核不合格\n补考\n重修"})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()})()

    assert rewrite_query(client, "挂科了咋办？") == ["挂科了咋办？", "考核不合格", "补考"]
