"""查询改写模块：将用户简短查询扩展为多个子查询，提升召回覆盖率。"""

from __future__ import annotations

from zhipuai import ZhipuAI

REWRITE_PROMPT = """你是一个查询改写助手。用户会在校园知识库中搜索学生手册相关问题。
请将用户的原始问题改写为 2-4 个不同角度的子查询，帮助检索到更全面的信息。

规则：
1. 保留原始问题的核心意图
2. 从不同角度/关键词重新表述
3. 每个子查询单独一行
4. 不要编号，不要多余解释
5. 第一行必须是原始问题本身

原始问题：{question}"""


def rewrite_query(client: ZhipuAI, question: str) -> list[str]:
    """用 LLM 将单个问题改写为多个子查询。"""
    prompt = REWRITE_PROMPT.format(question=question)
    response = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
        thinking={"type": "disabled"},
    )
    text = response.choices[0].message.content or ""
    queries = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not queries:
        return [question]
    return queries
