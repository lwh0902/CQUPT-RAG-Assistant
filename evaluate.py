"""
异步评测主模块。

这里使用 asyncio + to_thread + Semaphore(3)，避免评测时一题一题串行等待。
"""

import asyncio
import json
import logging
import time

from rag import ask_question_async, get_glm_client, init_rag_system_async
from settings import MAX_CONCURRENCY


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_test_cases(file_path="test_cases.json") -> list[dict]:
    """
    读取独立测试集文件。
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


test_cases = load_test_cases()


def evaluate_with_llm(client, case: dict, actual_answer: str) -> bool:
    """
    让大模型当裁判，判断回答是否命中预期要求。

    这里除了 expected 之外，还支持 must_include 和 must_not_include，
    让评测更适合学生手册这种规则型文档。
    """
    expected = case["expected"]
    must_include = case.get("must_include", [])
    must_not_include = case.get("must_not_include", [])
    case_type = case.get("type", "fact")

    if case_type == "refusal" and "无法确定" in actual_answer:
        return True

    judge_prompt = f"""
你是一名严格的 RAG 评测裁判。请判断下面这条回答是否通过测试。

【问题】
{case["question"]}

【题型】
{case_type}

【期望核心答案】
{expected}

【回答中必须包含的要点】
{must_include}

【回答中不应该出现的内容】
{must_not_include}

【实际回答】
{actual_answer}

判定规则：
1. 如果实际回答准确表达了期望核心答案，可以判定通过。
2. 如果 must_include 中的重要点缺失，判定不通过。
3. 如果 must_not_include 中的内容明显出现在回答里，判定不通过。
4. 对 refusal 题，如果文档中没有答案，回答“无法确定”应视为通过。

请只回答 YES 或 NO。
"""

    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.01,
    )
    content = response.choices[0].message.content.strip().upper()
    return "YES" in content


async def evaluate_with_llm_async(client, case: dict, actual_answer: str) -> bool:
    """
    把同步裁判调用放到线程里执行，避免阻塞事件循环。
    """
    return await asyncio.to_thread(evaluate_with_llm, client, case, actual_answer)


async def process_single_case(index: int, case: dict, retriever, client, semaphore: asyncio.Semaphore) -> dict:
    """
    异步处理单个测试样例。
    """
    started_at = time.perf_counter()

    async with semaphore:
        try:
            answer, _, pages = await ask_question_async(case["question"], retriever)
            passed = await evaluate_with_llm_async(client, case, answer)
            duration = time.perf_counter() - started_at

            result = {
                "index": index,
                "question": case["question"],
                "type": case.get("type", "fact"),
                "passed": passed,
                "answer": answer,
                "expected": case["expected"],
                "pages": pages,
                "duration": duration,
            }
            return result
        except Exception as exc:
            duration = time.perf_counter() - started_at
            return {
                "index": index,
                "question": case["question"],
                "type": case.get("type", "fact"),
                "passed": False,
                "answer": f"评测异常：{exc}",
                "expected": case["expected"],
                "pages": [],
                "duration": duration,
            }


def summarize_results(results: list[dict]) -> dict:
    """
    汇总总正确率和分题型正确率。
    """
    total = len(results)
    passed = sum(1 for item in results if item["passed"])

    type_summary = {}
    for item in results:
        case_type = item["type"]
        if case_type not in type_summary:
            type_summary[case_type] = {"total": 0, "passed": 0}

        type_summary[case_type]["total"] += 1
        if item["passed"]:
            type_summary[case_type]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "accuracy": (passed / total * 100) if total else 0.0,
        "type_summary": type_summary,
    }


async def run_tests_async():
    """
    并发执行整套评测。
    """
    logging.info("开始基于学生手册 PDF 的异步评测...")
    started_at = time.perf_counter()

    retriever, init_info = await init_rag_system_async()
    logging.info(init_info["message"])

    client = get_glm_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        asyncio.create_task(process_single_case(index, case, retriever, client, semaphore))
        for index, case in enumerate(test_cases, start=1)
    ]
    results = await asyncio.gather(*tasks)

    for item in results:
        status = "✅" if item["passed"] else "❌"
        logging.info(
            "%s [%s] %s | 题型: %s | 页码: %s | 耗时: %.2fs",
            status,
            item["index"],
            item["question"],
            item["type"],
            item["pages"] or "无",
            item["duration"],
        )
        if not item["passed"]:
            logging.warning("   期望: %s", item["expected"])
            logging.warning("   实际: %s", item["answer"])

    summary = summarize_results(results)
    total_duration = time.perf_counter() - started_at

    logging.info("=" * 50)
    logging.info("总正确率：%s/%s (%.1f%%)", summary["passed"], summary["total"], summary["accuracy"])
    for case_type, stats in summary["type_summary"].items():
        accuracy = stats["passed"] / stats["total"] * 100
        logging.info("题型[%s]：%s/%s (%.1f%%)", case_type, stats["passed"], stats["total"], accuracy)
    logging.info("并发上限：%s", MAX_CONCURRENCY)
    logging.info("总耗时：%.2fs", total_duration)
    logging.info("=" * 50)

    return results, summary


if __name__ == "__main__":
    asyncio.run(run_tests_async())
