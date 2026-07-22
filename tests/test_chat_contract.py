from routers.chat import ChatRequest, _sse_pack, build_system_prompt


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
            "long_term": "",
            "overflow_summary": "",
        }
    )

    assert "https://cqupt.edu.cn/notice/1" in prompt
    assert "不得把自身常识伪装成联网检索结果" in prompt
    assert "【近期对话摘要】" not in prompt
    # Anti-hedging rule: explicit numeric/clause evidence must be stated directly.
    assert "必须直接给出" in prompt
    assert "实际执行为准" in prompt


def test_sse_pack_uses_event_stream_data_prefix() -> None:
    packed = _sse_pack({"type": "delta", "content": "你好"})
    assert packed.startswith("data: ")
    assert packed.endswith("\n\n")
    assert '"type": "delta"' in packed or '"type":"delta"' in packed


def test_prompt_english_instruction_when_lang_en() -> None:
    prompt = build_system_prompt(
        {"knowledge": "手册条文", "web": "无"},
        lang="en-US",
    )
    assert "请用 English 回答用户" in prompt
    assert "保留中文原名" in prompt

    prompt_zh = build_system_prompt({"knowledge": "手册条文", "web": "无"})
    assert "请用 English 回答用户" not in prompt_zh


def test_retrieval_refusal_bilingual() -> None:
    from routers.chat import build_retrieval_refusal

    en = build_retrieval_refusal("insufficient", "en-US")
    assert "evidence is not strong enough" in en
    zh = build_retrieval_refusal("insufficient", "zh-CN")
    assert "证据不足以支持" in zh


def test_lang_validator_falls_back_to_zh() -> None:
    request = ChatRequest(
        session_id="session-12345",
        new_message="奖学金政策是什么？",
        lang="fr-FR",
    )
    assert request.lang == "zh-CN"
