"""Campus query normalization: aliases, keyword extraction, lexical expand."""

from __future__ import annotations

import re
from functools import lru_cache

# Longer / more specific aliases first when matching.
POLICY_ALIASES: list[tuple[str, list[str]]] = [
    ("国家励志奖学金", ["国家励志奖学金", "国励"]),
    ("国家奖学金", ["国家奖学金", "国奖"]),
    ("国家助学金", ["国家助学金", "助学金"]),
    ("学业奖学金", ["学业奖学金", "校内奖学金"]),
    ("社会奖学金", ["社会奖学金"]),
    ("郭长波奖学金", ["郭长波奖学金", "郭长波"]),
    ("科创文体奖学金", ["科创文体奖学金"]),
    ("综合素质测评", ["综合素质测评", "综测"]),
    ("卫生寝室", ["卫生寝室"]),
    ("五星文明寝室", ["五星文明寝室", "五星寝室"]),
    ("五星文明楼", ["五星文明楼"]),
    ("考核不合格", ["考核不合格", "挂科", "没过", "不及格", "一门课没过"]),
    ("补考", ["补考"]),
    ("重修", ["重修"]),
    ("休学", ["休学"]),
    ("复学", ["复学", "回来读", "返校复学"]),
    ("退学", ["退学", "没学上", "取消学籍"]),
    ("转学", ["转学"]),
    ("转专业", ["转专业"]),
    ("提前毕业", ["提前毕业", "早一年毕业", "提前修完"]),
    ("旷课", ["旷课", "不去上课", "没去上课", "缺勤"]),
    ("请假", ["请假"]),
    ("报到", ["报到", "入学手续", "按时报到"]),
    ("注册", ["注册", "暂缓注册"]),
    ("互斥", ["互斥", "不能同时", "不能一起拿", "一起拿不", "兼得"]),
    ("兼得", ["兼得", "同时获得", "同时申请"]),
    ("家庭经济困难", ["家庭经济困难", "困难生"]),
    ("处分", ["处分"]),
    ("申诉", ["申诉"]),
    ("学籍", ["学籍"]),
    ("毕业", ["毕业"]),
    ("学位", ["学位"]),
    ("奖学金", ["奖学金"]),
    # Dorm / daily-life policy terms (general lexicon, not topic routers).
    ("晚归", ["晚归", "晚点回寝", "晚点回宿舍", "回来太晚"]),
    ("夜不归宿", ["夜不归宿", "彻夜未归", "整晚不回", "过夜不归", "不回寝", "夜不归寝", "晚上不回", "不回"]),
    # Deduction phrasing: users ask 扣多少分/扣几分; the manual says 予以扣分.
    ("扣分", ["扣分", "扣多少分", "扣几分", "扣多少", "怎么扣"]),
    ("归寝", ["归寝", "回寝", "按时回宿舍", "回宿舍"]),
    ("学生公寓", ["学生公寓", "宿舍", "寝室", "公寓"]),
]

# Canonical policy terms kept for direct hit extraction / ranking.
POLICY_TERMS: list[str] = sorted(
    {canonical for canonical, _ in POLICY_ALIASES} | {
        "一等奖", "二等奖", "三等奖", "智育", "申请条件", "奖励标准",
        "评选条件", "考勤", "学科竞赛", "科研成果",
        "惩罚", "扣分", "门禁", "违纪",
    },
    key=len,
    reverse=True,
)

# Generic function words / question shells — never used alone as retrieval keywords.
_STOPWORDS: set[str] = {
    "什么", "怎么", "怎样", "如何", "是否", "可以", "能不能", "会不会",
    "一下", "这个", "那个", "我们", "你们", "他们", "自己", "还有",
    "一个", "一些", "不是", "就是", "以及", "或者", "如果", "因为",
    "所以", "但是", "然后", "已经", "还是", "的话", "请问", "老师",
    "学校", "重邮", "同学", "有没有", "到底", "一般", "相关", "进行",
    "问题", "情况", "时候", "地方", "东西", "这样", "那样", "这么",
    "那么", "哪些", "哪种", "多少", "几个", "为何", "为什么", "咋办",
    "怎么办", "会有", "没有", "不是", "一下", "之类",
    # Scaffolding tokens from follow-up resolution text ("用户追问：...").
    "用户", "追问", "原问题",
    # Generic scope words that appear all over the corpus and only add noise.
    "所有", "一切", "全部", "各种", "啥", "咋样", "怎么样",
    # Deictic / manner adverbs common in follow-ups; topic words carry retrieval.
    "直接", "立即", "马上", "然后",
}


def _alias_hits(question: str) -> list[str]:
    text = question or ""
    hits: list[str] = []
    seen: set[str] = set()
    for canonical, aliases in POLICY_ALIASES:
        if any(alias and alias in text for alias in aliases):
            if canonical not in seen:
                hits.append(canonical)
                seen.add(canonical)
    return hits


def _content_tokens(question: str, *, limit: int = 8) -> list[str]:
    """Fallback content words via jieba when the closed policy lexicon misses."""
    text = (question or "").strip()
    if not text:
        return []
    try:
        import jieba
    except Exception:
        return []

    _ensure_jieba_user_dict()
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in jieba.lcut(text):
        token = (raw or "").strip()
        if len(token) < 2 or len(token) > 12:
            continue
        if token in _STOPWORDS or token in seen:
            continue
        if re.fullmatch(r"[\d\.％%]+", token):
            continue
        # Skip pure punctuation / latin filler.
        if re.fullmatch(r"[\W_]+", token, flags=re.UNICODE):
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def extract_query_keywords(question: str, *, limit: int = 8) -> list[str]:
    """Extract retrieval/rerank keywords; never return whole-sentence fragments.

    Order:
    1) campus alias / policy lexicon hits (highest value)
    2) content tokens from the question (prevents empty-keyword collapse on
       out-of-lexicon but in-corpus asks such as 晚归)
    """
    text = (question or "").strip()
    if not text:
        return []

    keywords = _alias_hits(text)

    for term in POLICY_TERMS:
        if term in text and term not in keywords:
            keywords.append(term)

    # Always try to keep a few content tokens so rerank/evidence-gate still have
    # overlap signal when the closed lexicon is incomplete.
    for token in _content_tokens(text, limit=limit):
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break

    return keywords[:limit]


def lexical_expand_query(question: str) -> str:
    """Append formal policy terms for BM25/vector query without dropping the original."""
    text = (question or "").strip()
    if not text:
        return text
    extras = [term for term in extract_query_keywords(text) if term not in text]
    if not extras:
        return text
    # Keep expansion compact for embedding/BM25.
    return f"{text} {' '.join(extras[:4])}".strip()


@lru_cache(maxsize=1)
def alias_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for canonical, aliases in POLICY_ALIASES:
        for alias in aliases:
            table[alias] = canonical
    return table


def _ensure_jieba_user_dict() -> None:
    """Register campus terms so jieba keeps policy phrases intact."""
    try:
        import jieba
    except Exception:
        return
    for term in POLICY_TERMS:
        jieba.add_word(term)
    for canonical, aliases in POLICY_ALIASES:
        jieba.add_word(canonical)
        for alias in aliases:
            if alias:
                jieba.add_word(alias)


_ensure_jieba_user_dict()
