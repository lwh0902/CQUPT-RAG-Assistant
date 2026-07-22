from rag import extract_query_keywords, lexical_expand_query, rerank_documents
from langchain_core.documents import Document


def test_extract_keywords_maps_colloquial_aliases_to_policy_terms() -> None:
    assert "国家奖学金" in extract_query_keywords("国奖到底给多少钱，还能和国励一起拿不？")
    assert "国家励志奖学金" in extract_query_keywords("国奖到底给多少钱，还能和国励一起拿不？")
    assert "补考" in extract_query_keywords("我一门课没过，是补考还是重修？") or "考核不合格" in extract_query_keywords(
        "我一门课没过，是补考还是重修？"
    )
    assert "卫生寝室" in extract_query_keywords("寝室想评卫生寝室，检查分至少得多少？")


def test_extract_keywords_does_not_return_whole_sentence_fragments() -> None:
    keywords = extract_query_keywords("国奖到底给多少钱，要满足啥条件，还能和国励一起拿不？")
    assert keywords
    assert all(len(item) <= 12 for item in keywords)
    assert "国奖到底给多少钱" not in keywords


def test_extract_keywords_keeps_content_words_outside_closed_lexicon() -> None:
    """Closed lexicon must not collapse ordinary campus asks to empty keywords."""
    keywords = extract_query_keywords("宿舍晚归会有什么惩罚")
    assert keywords
    assert "晚归" in keywords
    # Should still avoid whole-sentence fragments.
    assert all(len(item) <= 12 for item in keywords)
    assert "宿舍晚归会有什么惩罚" not in keywords


def test_lexical_expand_query_appends_formal_terms() -> None:
    expanded = lexical_expand_query("国奖能和国励一起拿吗")
    assert "国家奖学金" in expanded
    assert "国家励志奖学金" in expanded
    assert expanded.startswith("国奖能和国励一起拿吗")


def test_rerank_prefers_document_with_policy_terms_from_colloquial_query() -> None:
    weak = Document(
        page_content="国家励志奖学金用于资助家庭经济困难学生。",
        metadata={"document_id": "manual", "page": 83},
    )
    strong = Document(
        page_content="本科生国家奖学金实施细则。国家奖学金用于奖励表现优异学生，不能同时获得国家励志奖学金。",
        metadata={"document_id": "manual", "page": 80},
    )
    ranked = rerank_documents("国奖到底给多少钱，还能和国励一起拿不？", [weak, strong])
    assert ranked[0].metadata["page"] == 80


def test_rerank_surfaces_penalty_page_for_late_return_query() -> None:
    definition = Document(
        page_content="未按时归寝记为晚归，彻夜未归记为夜不归宿，按公寓管理办法处理。",
        metadata={"document_id": "manual", "page": 138},
    )
    penalty = Document(
        page_content="晚归的，扣5分/人次；夜不归宿的，扣20分/人次。",
        metadata={"document_id": "manual", "page": 145},
    )
    noise = Document(
        page_content="课堂出勤作为评奖评优依据。",
        metadata={"document_id": "manual", "page": 134},
    )
    ranked = rerank_documents("宿舍晚归会有什么惩罚", [noise, definition, penalty])
    assert ranked[0].metadata["page"] in {138, 145}
    assert {doc.metadata["page"] for doc in ranked[:2]} >= {138, 145} or ranked[0].metadata["page"] == 145


def test_followup_scaffolding_tokens_are_not_keywords() -> None:
    """Resolved follow-up text must not leak 用户/追问 into retrieval keywords."""
    keywords = extract_query_keywords("晚归有什么惩罚\n用户追问：扣多少分")
    assert "用户" not in keywords
    assert "追问" not in keywords
    assert "晚归" in keywords


def test_generic_scope_words_are_not_keywords() -> None:
    keywords = extract_query_keywords("所有晚归有什么后果呀")
    assert "所有" not in keywords
    assert "直接" not in extract_query_keywords("如果直接不回呢")
    assert "晚归" in keywords


def test_deictic_no_return_maps_to_overnight_absence() -> None:
    keywords = extract_query_keywords("如果直接不回呢")
    assert "夜不归宿" in keywords


def test_rerank_surfaces_penalty_page_for_consequence_query() -> None:
    """后果-phrased questions must also trigger the sanction bonus."""
    consequence_only = Document(
        page_content="学生公寓管理服务中心每天将门禁刷卡异常记录整理后通报到相关学院。",
        metadata={"document_id": "manual", "page": 138},
    )
    penalty = Document(
        page_content="违反公寓区管理规定的：晚归的，扣5分/人次；夜不归宿的，扣20分/人次。",
        metadata={"document_id": "manual", "page": 145},
    )
    ranked = rerank_documents("所有晚归有什么后果呀", [consequence_only, penalty])
    assert ranked[0].metadata["page"] == 145


def test_format_context_respects_char_budget_but_keeps_first_doc() -> None:
    from rag import format_context

    docs = [
        Document(page_content="x" * 800, metadata={"document_id": "m", "page": 1}),
        Document(page_content="y" * 800, metadata={"document_id": "m", "page": 2}),
        Document(page_content="z" * 800, metadata={"document_id": "m", "page": 3}),
    ]
    ctx, pages = format_context(docs, max_chars=1000)
    assert pages == [1]  # budget exceeded after first doc
    assert "x" * 100 in ctx
    assert "y" * 100 not in ctx

    ctx_all, pages_all = format_context(docs, max_chars=10000)
    assert pages_all == [1, 2, 3]
