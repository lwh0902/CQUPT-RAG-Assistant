from services.article_chunker import parse_articles_from_pages, split_parent_into_children


def test_parse_articles_keeps_article_boundaries_and_titles() -> None:
    pages = [
        {
            "page": 76,
            "text": (
                "重庆邮电大学本科生公寓评优实施办法\n"
                "第三章 评优条件\n"
                "第四条 “卫生寝室”评选条件\n"
                "每学期学生寝室安全卫生检查平均分不低于90分。\n"
                "第五条 “五星文明寝室”评选条件\n"
                "无挂科记录，且卫生检查合格。"
            ),
        },
        {
            "page": 77,
            "text": (
                "第六条 “五星文明楼”评选条件\n"
                "楼栋卫生寝室率高。\n"
                "第十一条 “五星文明寝室”每学年评定一次，具体程序为：\n"
                "（一）自主申报。"
            ),
        },
    ]

    articles = parse_articles_from_pages(
        pages,
        document_id="student_manual_education_2025",
        document_name="学生手册（教育管理篇）2025版",
    )

    assert [item["article_no"] for item in articles] == [4, 5, 6, 11]
    art4 = articles[0]
    assert art4["article_title"] == "“卫生寝室”评选条件"
    assert "平均分不低于90分" in art4["text"]
    assert art4["page_start"] == 76
    assert art4["chapter"] == "评优条件"
    assert "五星文明楼" not in art4["text"]
    # Body-like first sentence should not become a fake title.
    art_bodyish = next(item for item in articles if item["article_no"] == 6)
    assert "楼栋环境好" not in (art_bodyish.get("article_title") or "")


def test_split_parent_into_children_preserves_parent_id() -> None:
    parent = {
        "article_id": "manual:art:4",
        "text": "第四条 条件如下：（一）分数达标；（二）无违纪。（三）按时申报。",
        "page_start": 76,
        "page_end": 76,
    }
    children = split_parent_into_children(parent, max_chars=24)
    assert len(children) >= 2
    assert all(child["parent_id"] == "manual:art:4" for child in children)
    assert "".join(child["text"] for child in children).replace(" ", "")  # non-empty join


def test_merge_short_articles_builds_larger_parents_with_article_children() -> None:
    from services.article_chunker import merge_articles_into_parents

    short_a = "第一条 为规范评优工作，制定本条件。" + ("说明。" * 8)
    short_b = "第四条 “卫生寝室”评选条件。" + ("平均分不低于八十五分。" * 6)
    short_c = "第五条 “五星文明寝室”评选条件。" + ("成员无挂科记录。" * 7)
    long_text = "第二十条 详细规则。" + ("补充条款内容。" * 40)
    articles = [
        {
            "article_id": "m:1",
            "document_id": "manual",
            "document_name": "手册",
            "article_no": 1,
            "article_title": "",
            "chapter_no": 3,
            "chapter": "评优条件",
            "section_no": None,
            "section": "",
            "page_start": 76,
            "page_end": 76,
            "char_count": len(short_a),
            "text": short_a,
        },
        {
            "article_id": "m:4",
            "document_id": "manual",
            "document_name": "手册",
            "article_no": 4,
            "article_title": "“卫生寝室”评选条件",
            "chapter_no": 3,
            "chapter": "评优条件",
            "section_no": None,
            "section": "",
            "page_start": 76,
            "page_end": 76,
            "char_count": len(short_b),
            "text": short_b,
        },
        {
            "article_id": "m:5",
            "document_id": "manual",
            "document_name": "手册",
            "article_no": 5,
            "article_title": "“五星文明寝室”评选条件",
            "chapter_no": 3,
            "chapter": "评优条件",
            "section_no": None,
            "section": "",
            "page_start": 76,
            "page_end": 77,
            "char_count": len(short_c),
            "text": short_c,
        },
        {
            "article_id": "m:20",
            "document_id": "manual",
            "document_name": "手册",
            "article_no": 20,
            "article_title": "很长的一条",
            "chapter_no": 4,
            "chapter": "其他",
            "section_no": None,
            "section": "",
            "page_start": 90,
            "page_end": 91,
            "char_count": len(long_text),
            "text": long_text,
        },
    ]

    parents, children = merge_articles_into_parents(
        articles,
        min_parent_chars=200,
        max_parent_chars=900,
    )

    assert len(parents) == 2
    merged = next(p for p in parents if p["merge_count"] > 1)
    solo = next(p for p in parents if p["merge_count"] == 1)
    assert merged["char_count"] >= 200
    assert "卫生寝室" in merged["text"] and "五星文明寝室" in merged["text"]
    assert solo["article_nos"] == [20]
    # Children remain original articles for retrieval.
    assert {c["source_article_id"] for c in children} == {"m:1", "m:4", "m:5", "m:20"}
    assert all(c["parent_id"] for c in children)
