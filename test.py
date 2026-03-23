"""
简单本地自测脚本。

这个文件保留为轻量 smoke test：
1. 初始化学生手册索引
2. 跑少量样例
3. 快速确认主链路是否正常
"""

from rag import ask_question, init_rag_system


SMOKE_TEST_CASES = [
    ("国家奖学金的奖励标准是多少？", "10000元"),
    ("学生未经批准连续多久未参加教学活动，学校可予退学处理？", "连续两周"),
    ("学生手册里有写图书馆周末几点关门吗？", "无法确定"),
]


def run_smoke_tests():
    """
    运行最小冒烟测试。
    """
    print("开始初始化学生手册问答系统...")
    retriever, init_info = init_rag_system()
    print(init_info["message"])

    correct = 0
    total = len(SMOKE_TEST_CASES)

    for index, (question, expected) in enumerate(SMOKE_TEST_CASES, start=1):
        answer, _, pages = ask_question(question, retriever)
        passed = expected in answer
        status = "✅" if passed else "❌"

        print(f"{status} [{index}/{total}] 问题：{question}")
        print(f"   命中页码：{pages or '无'}")
        print(f"   回答：{answer}")

        if passed:
            correct += 1

    print(f"\n冒烟测试完成：{correct}/{total} ({correct / total * 100:.1f}%)")


if __name__ == "__main__":
    run_smoke_tests()
