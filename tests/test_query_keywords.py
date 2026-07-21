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
