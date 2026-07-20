from services.confidence import calculate_confidence
from services.evidence import EvidenceSource, deduplicate_evidence, normalize_evidence


def test_web_evidence_without_real_http_url_is_rejected() -> None:
    source = normalize_evidence(
        {
            "id": "web-1",
            "source_type": "web",
            "title": "政策通知",
            "snippet": "这是一段摘要",
            "url": "not-a-url",
        }
    )

    assert source is None


def test_deduplicate_evidence_keeps_stronger_copy_of_same_url() -> None:
    weaker = EvidenceSource(
        id="web-1",
        source_type="web",
        title="研究生政策",
        snippet="摘要",
        url="https://example.edu.cn/policy/",
        relevance_score=0.5,
        authority_score=0.6,
    )
    stronger = EvidenceSource(
        id="web-2",
        source_type="web",
        title="研究生政策",
        snippet="更完整的摘要",
        url="https://example.edu.cn/policy",
        relevance_score=0.9,
        authority_score=0.9,
    )

    assert deduplicate_evidence([weaker, stronger]) == [stronger]


def test_high_authority_consistent_evidence_is_high_confidence() -> None:
    evidence = [
        EvidenceSource(
            id="kb-1",
            source_type="knowledge_base",
            title="学生手册",
            snippet="国家奖学金奖励标准为 10000 元。",
            document_name="学生手册.pdf",
            page=12,
            relevance_score=0.95,
            authority_score=0.98,
        ),
        EvidenceSource(
            id="web-1",
            source_type="web",
            title="奖助政策通知",
            snippet="国家奖学金奖励标准为 10000 元。",
            url="https://cqupt.edu.cn/policy",
            site_name="重庆邮电大学",
            relevance_score=0.9,
            authority_score=0.95,
        ),
    ]

    result = calculate_confidence(evidence)

    assert result.level == "high"
    assert result.score >= 0.8
    assert result.uncertain_points == []


def test_conflicting_evidence_is_never_high_confidence() -> None:
    evidence = [
        EvidenceSource(
            id="kb-1",
            source_type="knowledge_base",
            title="学生手册",
            snippet="国家奖学金奖励标准为 10000 元。",
            document_name="学生手册.pdf",
            page=12,
            relevance_score=0.95,
            authority_score=0.98,
        ),
        EvidenceSource(
            id="web-1",
            source_type="web",
            title="旧网页",
            snippet="国家奖学金奖励标准为 8000 元。",
            url="https://example.org/old-policy",
            relevance_score=0.9,
            authority_score=0.8,
        ),
    ]

    result = calculate_confidence(evidence)

    assert result.level in {"low", "unknown"}
    assert result.uncertain_points


def test_empty_evidence_is_unknown() -> None:
    result = calculate_confidence([])

    assert result.level == "unknown"
    assert result.score == 0.0
