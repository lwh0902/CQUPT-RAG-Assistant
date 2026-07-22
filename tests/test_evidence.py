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

    assert result.level in {"low", "medium", "unknown"}
    assert result.uncertain_points
    assert result.level != "high"


def test_same_manual_neighbor_pages_are_not_numeric_conflicts() -> None:
    """Late-return pages mix 5分/20分/23:30 — that is not a source conflict."""
    evidence = [
        EvidenceSource(
            id="kb-138",
            source_type="knowledge_base",
            title="学生手册",
            snippet="学生应当在每天早晨6:00到23:30期间凭校园一卡通刷卡进出。未按时归寝记为晚归。",
            document_id="student_manual_education_2025",
            document_name="学生手册（教育管理篇）2025版",
            page=138,
            relevance_score=0.4,
            authority_score=0.95,
        ),
        EvidenceSource(
            id="kb-145",
            source_type="knowledge_base",
            title="学生手册",
            snippet="晚归的，扣5分/人次；夜不归宿的，扣20分/人次。",
            document_id="student_manual_education_2025",
            document_name="学生手册（教育管理篇）2025版",
            page=145,
            relevance_score=0.4,
            authority_score=0.95,
        ),
        EvidenceSource(
            id="kb-144",
            source_type="knowledge_base",
            title="学生手册",
            snippet="存在违规行为（卫生不合格、晚归等违规行为除外）的，工作人员将出具积分处罚告知书。",
            document_id="student_manual_education_2025",
            document_name="学生手册（教育管理篇）2025版",
            page=144,
            relevance_score=0.35,
            authority_score=0.95,
        ),
    ]

    result = calculate_confidence(evidence)

    assert "冲突" not in "".join(result.uncertain_points)
    assert result.level in {"high", "medium"}
    assert result.level != "unknown"


def test_empty_evidence_is_unknown() -> None:
    result = calculate_confidence([])

    assert result.level == "unknown"
    assert result.score == 0.0


def test_same_manual_reward_rows_are_not_numeric_conflicts() -> None:
    """Distinct quoted reward rows (卫生寝室 5分 / 五星文明寝室 10分 / 文明寝室 20分)
    on neighbor pages must not collapse into one metric key."""
    p144 = (
        "（1）荣获学校“卫生寝室”荣誉的寝室成员加5分，寝室长另加3分；"
        "（2）荣获学校“五星文明寝室”荣誉的寝室成员加10分，寝室长另加3分；"
    )
    p145 = (
        "（3）荣获重庆市“文明寝室”荣誉的寝室成员加20分，寝室长另加6分；"
        "（1）晚归的，扣5分/人次；夜不归宿的，扣20分/人次；"
    )
    evidence = [
        EvidenceSource(
            id="kb-145",
            source_type="knowledge_base",
            title="学生手册（教育管理篇）2025版",
            snippet=p145,
            document_id="student_manual_education_2025",
            document_name="学生手册（教育管理篇）2025版",
            page=145,
            relevance_score=0.6,
            authority_score=0.95,
        ),
        EvidenceSource(
            id="kb-144",
            source_type="knowledge_base",
            title="学生手册（教育管理篇）2025版",
            snippet=p144,
            document_id="student_manual_education_2025",
            document_name="学生手册（教育管理篇）2025版",
            page=144,
            relevance_score=0.55,
            authority_score=0.95,
        ),
    ]

    result = calculate_confidence(evidence)

    assert "冲突" not in "".join(result.uncertain_points)
    assert result.level in {"high", "medium"}


def test_real_metric_conflict_still_flagged_same_manual() -> None:
    """Same metric stated with different values must still warn."""
    evidence = [
        EvidenceSource(
            id="kb-a",
            source_type="knowledge_base",
            title="奖学金办法",
            snippet="国家奖学金奖励标准为10000元。",
            document_id="rules_2025",
            document_name="本科生奖学金评定办法",
            page=3,
            relevance_score=0.9,
            authority_score=0.95,
        ),
        EvidenceSource(
            id="kb-b",
            source_type="knowledge_base",
            title="奖学金办法",
            snippet="国家奖学金奖励标准为8000元。",
            document_id="rules_2024",
            document_name="本科生奖学金评定办法（旧版）",
            page=3,
            relevance_score=0.8,
            authority_score=0.9,
        ),
    ]

    result = calculate_confidence(evidence)

    assert "冲突" in "".join(result.uncertain_points)
    assert result.level != "high"
