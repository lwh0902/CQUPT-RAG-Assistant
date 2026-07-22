from langchain_core.documents import Document

from services.parent_child_index import (
    build_parent_child_corpus,
    expand_children_to_parents,
    expand_neighbor_pages,
)


def test_build_parent_child_corpus_indexes_children_with_parent_ids() -> None:
    pages = [
        Document(
            page_content=(
                "第三章 评优条件\n"
                "第四条 “卫生寝室”评选条件\n"
                "每学期学生寝室安全卫生检查平均分在85分以上。\n"
                "第五条 “五星文明寝室”评选条件\n"
                "无挂科记录。"
            ),
            metadata={
                "document_id": "manual",
                "document_name": "手册",
                "document_type": "manual",
                "topic": "dorm",
                "authority_level": 90,
                "page": 76,
                "source": "x.pdf",
                "file_path": "x.pdf",
            },
        )
    ]
    children, parents = build_parent_child_corpus(pages, min_parent_chars=50, max_parent_chars=900)
    assert children
    assert all(doc.metadata.get("chunk_type") == "child" for doc in children)
    assert all(doc.metadata.get("parent_id") in parents for doc in children)
    assert any("卫生寝室" in doc.page_content for doc in children)


def test_build_parent_child_keeps_non_article_front_matter_pages() -> None:
    pages = [
        Document(
            page_content="重邮精神文化\n重邮校训\n修德\n博学\n求实\n创新\n重邮精神\n敢为人先",
            metadata={
                "document_id": "manual",
                "document_name": "手册",
                "document_type": "manual",
                "topic": "culture",
                "authority_level": 70,
                "page": 1,
                "source": "x.pdf",
                "file_path": "x.pdf",
            },
        ),
        Document(
            page_content=(
                "第一章 总则\n"
                "第一条 为规范学生管理，制定本规定。\n"
                "第二条 本办法适用于全日制本科生。"
            ),
            metadata={
                "document_id": "manual",
                "document_name": "手册",
                "document_type": "manual",
                "topic": "rules",
                "authority_level": 70,
                "page": 7,
                "source": "x.pdf",
                "file_path": "x.pdf",
            },
        ),
    ]
    children, parents = build_parent_child_corpus(pages, min_parent_chars=50, max_parent_chars=900)
    assert any("校训" in doc.page_content for doc in children)
    assert any(doc.metadata.get("page") == 1 for doc in children)
    assert any(str(pid).endswith(":page:1") for pid in parents)
    # Dual lane: page children exist alongside article children.
    assert any(str(doc.metadata.get("child_id") or "").endswith(":full") for doc in children)
    assert any("第1条" in doc.page_content for doc in children)


def test_expand_children_to_parents_dedupes_by_parent() -> None:
    parent_store = {
        "p1": {
            "parent_id": "p1",
            "document_id": "manual",
            "document_name": "手册",
            "title": "评优条件 第4–5条",
            "text": "第四条 ...\n第五条 ...",
            "page_start": 76,
            "page_end": 77,
            "article_nos": [4, 5],
            "merge_count": 2,
            "authority_level": 90,
        }
    }
    c1 = Document(page_content="child4", metadata={"parent_id": "p1", "page": 76, "document_id": "manual"})
    c2 = Document(page_content="child5", metadata={"parent_id": "p1", "page": 77, "document_id": "manual"})
    expanded = expand_children_to_parents([c1, c2], parent_store=parent_store)
    assert len(expanded) == 1
    assert expanded[0].metadata["chunk_type"] == "parent"
    assert expanded[0].metadata["page"] == 76
    assert "第四条" in expanded[0].page_content


def test_expand_children_to_parents_dedupes_same_document_page_across_lanes() -> None:
    parent_store = {
        "manual:parent:1:76-76": {
            "parent_id": "manual:parent:1:76-76",
            "document_id": "manual",
            "document_name": "手册",
            "title": "第4条",
            "text": "第四条 卫生寝室条件。",
            "page_start": 76,
            "page_end": 76,
            "article_nos": [4],
            "merge_count": 1,
            "chunk_origin": "article",
            "authority_level": 90,
        },
        "manual:page:76": {
            "parent_id": "manual:page:76",
            "document_id": "manual",
            "document_name": "手册",
            "title": "第76页",
            "text": "整页：第四条 卫生寝室条件。第五条 ...",
            "page_start": 76,
            "page_end": 76,
            "article_nos": [],
            "merge_count": 1,
            "chunk_origin": "page",
            "authority_level": 90,
        },
    }
    article_hit = Document(
        page_content="child-article",
        metadata={"parent_id": "manual:parent:1:76-76", "page": 76, "document_id": "manual", "child_id": "a"},
    )
    page_hit = Document(
        page_content="child-page",
        metadata={"parent_id": "manual:page:76", "page": 76, "document_id": "manual", "child_id": "p"},
    )
    expanded = expand_children_to_parents([article_hit, page_hit], parent_store=parent_store)
    assert len(expanded) == 1
    assert expanded[0].metadata["page"] == 76
    assert expanded[0].metadata["parent_id"] == "manual:parent:1:76-76"


def test_expand_neighbor_pages_appends_adjacent_page_parents() -> None:
    parent_store = {
        "manual:page:11": {
            "parent_id": "manual:page:11",
            "document_id": "manual",
            "document_name": "手册",
            "title": "第11页",
            "text": "第十一条 考核与成绩。",
            "page_start": 11,
            "page_end": 11,
            "article_nos": [],
            "chunk_origin": "page",
            "authority_level": 70,
        },
        "manual:page:12": {
            "parent_id": "manual:page:12",
            "document_id": "manual",
            "document_name": "手册",
            "title": "第12页",
            "text": "第十二条 转学条件。",
            "page_start": 12,
            "page_end": 12,
            "article_nos": [],
            "chunk_origin": "page",
            "authority_level": 70,
        },
    }
    seed = Document(
        page_content="hit",
        metadata={"document_id": "manual", "page": 11, "parent_id": "manual:page:11"},
    )
    expanded = expand_neighbor_pages([seed], parent_store=parent_store, radius=1, max_seed_docs=3)
    pages = [doc.metadata.get("page") for doc in expanded]
    assert 11 in pages
    assert 12 in pages
    assert any(doc.metadata.get("neighbor_expanded") for doc in expanded if doc.metadata.get("page") == 12)


def test_expand_neighbor_pages_seed_zero_means_all_docs() -> None:
    """max_seed_docs<=0 must treat every doc as a seed (pool is bounded upstream)."""
    parent_store = {
        "manual:page:12": {
            "parent_id": "manual:page:12",
            "document_id": "manual",
            "document_name": "手册",
            "title": "第12页",
            "text": "第十二条 转学条件。",
            "page_start": 12,
            "page_end": 12,
            "article_nos": [],
            "chunk_origin": "page",
            "authority_level": 70,
        },
    }
    late_seed = Document(
        page_content="tail hit",
        metadata={"document_id": "manual", "page": 11, "parent_id": "manual:page:11"},
    )
    fillers = [
        Document(page_content=f"f{i}", metadata={"document_id": "manual", "page": 100 + i})
        for i in range(6)
    ]
    # late_seed sits beyond any fixed top-5 cap; with 0 it must still expand.
    expanded = expand_neighbor_pages(
        [*fillers, late_seed],
        parent_store=parent_store,
        radius=1,
        max_seed_docs=0,
    )
    assert 12 in [doc.metadata.get("page") for doc in expanded]
    capped = expand_neighbor_pages(
        [*fillers, late_seed],
        parent_store=parent_store,
        radius=1,
        max_seed_docs=5,
    )
    assert 12 not in [doc.metadata.get("page") for doc in capped]


def _page_parent(doc_id: str, page: int, text: str, name: str = "办法") -> dict:
    return {
        "parent_id": f"{doc_id}:page:{page}",
        "document_id": doc_id,
        "document_name": name,
        "title": f"第{page}页",
        "text": text,
        "page_start": page,
        "page_end": page,
        "article_nos": [],
        "chunk_origin": "page",
        "authority_level": 90,
    }


def test_extract_citation_titles_and_resolve() -> None:
    from services.parent_child_index import (
        extract_citation_titles,
        resolve_cited_document_ids,
    )

    titles = extract_citation_titles(
        "情节严重的，按照《重庆邮电大学学生违纪处分实施办法》处理；参照《学生手册》。"
    )
    assert "重庆邮电大学学生违纪处分实施办法" in titles
    assert resolve_cited_document_ids("重庆邮电大学学生违纪处分实施办法") == [
        "disciplinary_rules_2017"
    ]
    assert resolve_cited_document_ids("本科生社会奖学金评定办法") == [
        "social_scholarship_rules"
    ]
    assert resolve_cited_document_ids("不存在的办法") == []


def test_expand_cited_documents_pulls_cited_pages() -> None:
    from services.parent_child_index import expand_cited_documents

    store = {
        "student_manual_education_2025:page:138": _page_parent(
            "student_manual_education_2025",
            138,
            "未按时归寝记为晚归，情节严重的，按照《重庆邮电大学学生违纪处分实施办法》处理。",
            "学生手册",
        ),
        "disciplinary_rules_2017:page:11": _page_parent(
            "disciplinary_rules_2017", 11, "处分决定的送达与归档。", "处分办法"
        ),
        "disciplinary_rules_2017:page:12": _page_parent(
            "disciplinary_rules_2017", 12, "晚归屡教不改的，给予警告处分。", "处分办法"
        ),
    }
    seed = Document(
        page_content="未按时归寝记为晚归，情节严重的，按照《重庆邮电大学学生违纪处分实施办法》处理。",
        metadata={"document_id": "student_manual_education_2025", "page": 138},
    )
    expanded = expand_cited_documents(
        "晚归严重会受到什么处分",
        [seed],
        parent_store=store,
        max_docs=2,
        max_pages_per_doc=2,
    )
    cited = [d for d in expanded if d.metadata.get("cited_expanded")]
    assert cited, "cited document pages should be pulled in"
    assert all(d.metadata["document_id"] == "disciplinary_rules_2017" for d in cited)
    # 同时含“晚归+处分”的 P12 应排在只含“处分”的 P11 前
    assert cited[0].metadata["page"] == 12
    assert {d.metadata["page"] for d in cited} == {11, 12}


def test_expand_cited_documents_skips_doc_already_in_pool() -> None:
    from services.parent_child_index import expand_cited_documents

    store = {
        "student_manual_education_2025:page:138": _page_parent(
            "student_manual_education_2025",
            138,
            "按照《重庆邮电大学学生违纪处分实施办法》处理。",
            "学生手册",
        ),
        "disciplinary_rules_2017:page:11": _page_parent(
            "disciplinary_rules_2017", 11, "处分种类。", "处分办法"
        ),
    }
    seeds = [
        Document(
            page_content="按照《重庆邮电大学学生违纪处分实施办法》处理。",
            metadata={"document_id": "student_manual_education_2025", "page": 138},
        ),
        Document(
            page_content="处分种类。",
            metadata={"document_id": "disciplinary_rules_2017", "page": 11},
        ),
    ]
    expanded = expand_cited_documents(
        "处分种类", seeds, parent_store=store, max_docs=2, max_pages_per_doc=2
    )
    assert not any(d.metadata.get("cited_expanded") for d in expanded)


def test_expand_cited_documents_disabled_by_zero_cap() -> None:
    from services.parent_child_index import expand_cited_documents

    seed = Document(
        page_content="按照《重庆邮电大学学生违纪处分实施办法》处理。",
        metadata={"document_id": "student_manual_education_2025", "page": 138},
    )
    assert expand_cited_documents("处分", [seed], parent_store={}, max_docs=0) == [seed]
