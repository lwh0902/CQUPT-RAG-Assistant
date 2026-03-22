"""
自动化评测脚本：评估 RAG 问答效果（LLM-as-a-Judge 版本）
"""

import json
import logging
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

from rag import load_and_split_document, create_vector_db, ask_question


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_test_cases(file_path="test_cases.json"):
    """
    从独立 JSON 文件中读取测试集。

    JSON 中每一项格式为：
    {
      "question": "...",
      "expected": "..."
    }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到测试集文件：{file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    cases = []
    for item in raw_cases:
        question = item["question"]
        expected = item["expected"]
        cases.append((question, expected))
    return cases


# 保留这个变量名，兼容 async.py 等已有调用。
test_cases = load_test_cases()


def evaluate_with_llm(client: ZhipuAI, question: str, expected: str, actual: str) -> bool:
    """
    让大模型来当“裁判”，判断实际回答是否命中了期望答案。
    """
    if expected == "无法确定" and expected in actual:
        return True

    judge_prompt = f"""
    你是一个严格的评判员。请判断【实际回答】是否准确地包含了【预期核心事实】。
    用户问题: {question}
    预期核心事实: {expected}
    实际回答: {actual}

    请仔细对比。如果实际回答意思正确或包含了预期核心事实，请严格只回复大写字母 "YES"，否则回复 "NO"。
    """

    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.01,
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except Exception as exc:
        logging.error(f"调用裁判模型失败: {exc}")
        return False


def run_tests():
    """
    执行整套自动化评测。
    """
    logging.info("开始 RAG 效果自动化评测（LLM-as-a-Judge 模式）...")

    try:
        chunks = load_and_split_document("cqupt.txt")
        retriever = create_vector_db(chunks)
    except Exception as exc:
        logging.error(f"初始化 RAG 系统失败: {exc}")
        return

    load_dotenv()
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        logging.error("未找到 ZHIPU_API_KEY，请检查 .env 文件")
        return
    client = ZhipuAI(api_key=api_key)

    correct = 0
    total = len(test_cases)

    for i, (q, expected) in enumerate(test_cases, 1):
        try:
            actual_ans, _ = ask_question(q, retriever)
        except Exception as exc:
            logging.warning(f"❌ [{i}/{total}] 调用 RAG 失败 | 问题: {q} | 错误: {exc}")
            continue

        is_correct = evaluate_with_llm(client, q, expected, actual_ans)

        if is_correct:
            logging.info(f"✅ [{i}/{total}] 通过 | 问题: {q}")
            correct += 1
        else:
            logging.warning(f"❌ [{i}/{total}] 未通过 | 问题: {q}")
            logging.warning(f"   预期核心事实: {expected}")
            logging.warning(f"   实际回答: {actual_ans}")

    logging.info(f"测试完成！智能评估准确率: {correct}/{total} ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    run_tests()
