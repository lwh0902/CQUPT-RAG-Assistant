"""查询改写模块：将用户简短查询扩展为多个子查询，提升召回覆盖率。"""

from __future__ import annotations

from typing import Any

from settings import MODEL_NAME

REWRITE_PROMPT = """你是一个查询改写助手。用户会用口语在校园知识库中搜索学生手册相关问题。
请将用户的原始问题改写为最多 3 个可检索的子查询，帮助检索到更全面的信息。

规则：
1. 第一行必须原样保留用户问题。
2. 后续行把口语表达转换为学生手册中可能出现的正式制度术语；保留问题中的对象、条件和限制，不能自行添加事实。
3. 不要做机械同义词替换。只有确实存在口语与制度术语差异时才增加子查询；最多补 2 行。
4. 每个子查询单独一行，不要编号、解释或 Markdown。

以下示例用于理解“口语意图 -> 制度检索术语”的方式，不是必须逐字套用的映射表：
- 挂科 -> 考核不合格、补考、重修
- 缓交学费 -> 暂缓注册、助学贷款、资助
- 复学 -> 休学期满、复学申请、复查合格
- 国奖 -> 国家奖学金、申请条件、互斥规则
- 寝室评优 -> 卫生寝室、五星文明寝室、五星文明楼

原始问题：{question}"""


def rewrite_query(client: Any, question: str) -> list[str]:
    """用 LLM 将单个问题改写为多个子查询。"""
    prompt = REWRITE_PROMPT.format(question=question)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = response.choices[0].message.content or ""
    queries = [line.strip() for line in text.strip().splitlines() if line.strip()]
    expansions = [query for query in queries if query != question]
    return [question, *expansions[:2]]
