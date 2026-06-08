"""
对比实验脚本：依次运行四种检索策略，输出对比报告。

策略：
1. baseline  - 纯向量检索，无重排序
2. rerank    - 向量检索 + 关键词重排序
3. rewrite   - 查询改写 + 向量检索 + 关键词重排序
4. hybrid    - 查询改写 + 向量+BM25混合检索 + RRF融合排序
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from evaluate import (
    load_test_cases,
    evaluate_with_llm,
    summarize_results,
)
from rag import (
    get_glm_client,
    init_rag_system_async,
    ask_question_async,
    set_strategy,
    init_bm25_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STRATEGIES = ["baseline", "rerank", "rewrite", "hybrid"]


async def run_single_strategy(strategy: str, test_cases: list[dict]) -> dict:
    """运行单个策略的评测。"""
    set_strategy(strategy)
    logging.info("=" * 60)
    logging.info("开始评测策略: %s", strategy)
    logging.info("=" * 60)

    # hybrid 需要预构建 BM25 索引
    if strategy == "hybrid":
        logging.info("正在构建 BM25 索引...")
        init_bm25_index()

    retriever, _ = await init_rag_system_async()
    client = get_glm_client()

    results = []
    for index, case in enumerate(test_cases, start=1):
        started_at = time.perf_counter()
        try:
            answer, _, pages = await ask_question_async(case["question"], retriever)
            passed = await asyncio.to_thread(evaluate_with_llm, client, case, answer)
            duration = time.perf_counter() - started_at
            results.append({
                "index": index,
                "question": case["question"],
                "type": case.get("type", "fact"),
                "passed": passed,
                "pages": pages,
                "duration": duration,
            })
            status = "PASS" if passed else "FAIL"
            logging.info("  [%s] %s | %s | %.2fs", status, index, case["question"][:30], duration)
        except Exception as exc:
            duration = time.perf_counter() - started_at
            results.append({
                "index": index,
                "question": case["question"],
                "type": case.get("type", "fact"),
                "passed": False,
                "pages": [],
                "duration": duration,
            })
            logging.error("  [ERROR] %s | %s | %.2fs", index, case["question"][:30], duration)

    summary = summarize_results(results)
    summary["strategy"] = strategy
    summary["avg_duration"] = sum(r["duration"] for r in results) / len(results) if results else 0

    logging.info("策略 [%s] 完成: %s/%s (%.1f%%)", strategy, summary["passed"], summary["total"], summary["accuracy"])
    return summary


async def run_comparison():
    test_cases = load_test_cases()
    logging.info("测试集大小: %d 条", len(test_cases))

    all_summaries = []
    for strategy in STRATEGIES:
        summary = await run_single_strategy(strategy, test_cases)
        all_summaries.append(summary)

    # 输出对比报告
    print("\n" + "=" * 70)
    print("RAG 检索策略对比实验报告")
    print("=" * 70)
    print(f"{'策略':<12} {'准确率':<12} {'通过/总数':<12} {'平均耗时':<12}")
    print("-" * 70)
    for s in all_summaries:
        print(
            f"{s['strategy']:<12} "
            f"{s['accuracy']:.1f}%{'':<6} "
            f"{s['passed']}/{s['total']:<8} "
            f"{s['avg_duration']:.2f}s"
        )

    # 分题型对比
    print("\n分题型对比:")
    all_types = set()
    for s in all_summaries:
        all_types.update(s["type_summary"].keys())

    print(f"{'策略':<12}", end="")
    for t in sorted(all_types):
        print(f" {t:<16}", end="")
    print()
    print("-" * 70)
    for s in all_summaries:
        print(f"{s['strategy']:<12}", end="")
        for t in sorted(all_types):
            stats = s["type_summary"].get(t, {"passed": 0, "total": 0})
            acc = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
            print(f" {acc:.0f}%{'':<12}", end="")
        print()

    print("=" * 70)

    # 保存 JSON
    report_path = Path("experiment_report.json")
    report_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("详细报告已保存到 %s", report_path)

    return all_summaries


if __name__ == "__main__":
    asyncio.run(run_comparison())
