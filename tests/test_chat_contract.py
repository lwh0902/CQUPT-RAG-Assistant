from routers.chat import ChatRequest, build_system_prompt


def test_chat_request_disables_web_search_by_default() -> None:
    request = ChatRequest(
        session_id="session-12345",
        new_message="奖学金政策是什么？",
    )

    assert request.web_search_enabled is False
    assert request.user_id is None


def test_chat_prompt_includes_only_retrieved_web_evidence() -> None:
    prompt = build_system_prompt(
        {
            "knowledge": "【学生手册】\n奖学金资料",
            "web": "【学校官网｜https://cqupt.edu.cn/notice/1】\n最新通知",
            "long_term": "无",
            "short_term": "无",
        }
    )

    assert "https://cqupt.edu.cn/notice/1" in prompt
    assert "不得把自身常识伪装成联网检索结果" in prompt
