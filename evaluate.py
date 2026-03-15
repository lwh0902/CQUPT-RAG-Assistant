"""
自动化测试脚本 - 工业级 RAG 效果评估 (LLM-as-a-Judge)
文件名: evaluate_rag.py
"""
import os
import logging
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 从你的核心 RAG 文件中导入函数（确保你的核心文件名为 rag.py）
from rag import load_and_split_document, create_vector_db, ask_question

# ============ 1. 工业级日志配置 ============
# 抛弃 print，使用 logging。这样在服务器上跑的时候，日志可以存入文件，方便排查故障。
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============ 2. 测试用例数据 ============
# 真实场景下这些数据会放在 json 或 yaml 里，为了演示先放在这里
test_cases = [
    # 文档内问题（应能正确回答）
    ("重邮校训是什么？", "修德、博学、求实、创新"),
    ("图书馆周末几点关门？", "21:00"),
    ("国家奖学金多少钱？", "8000元"),
    ("校园卡补办要多少钱？", "15元"),
    ("心理中心怎么预约？", "拨打62461999或微信公众号"),
    ("快递点营业到几点？", "20:00"),
    ("校医院周末开吗？", "仅上午"),
    ("一食堂主打什么菜？", "川菜"),
    ("教务处电话多少？", "023-62461114"),
    ("宿舍有空调吗？", "有"),
    
    # 文档外问题（测试幻觉控制）
    ("重邮校长叫什么？", "无法确定"),
    ("2025年考研分数线？", "无法确定"),
    ("食堂有奶茶店吗？", "无法确定"),
    ("重邮有几个校区？", "无法确定"),
    ("计算机学院院长是谁？", "无法确定"),
    ("校园网密码怎么改？", "无法确定"),
    ("重邮建校多少年？", "无法确定"),
    ("游泳馆开放时间？", "无法确定"),
    ("重邮在哪个省？", "无法确定"),  # 实际能答，但测试模型是否依赖资料
    ("今天天气怎么样？", "无法确定")
]

# ============ 3. 大模型裁判逻辑 ============
def evaluate_with_llm(client: ZhipuAI, question: str, expected: str, actual: str) -> bool:
    """
    让大模型来当裁判，判断实际回答是否命中了预期核心事实。
    """
    # 如果预期是拒答，且模型确实拒答了，直接返回 True（提高效率，省一次 API 调用）
    if expected == "无法确定" and expected in actual:
        return True
        
    judge_prompt = f"""
    你是一个严苛的评判员。请判断【实际回答】是否准确地包含了【预期核心事实】。
    用户问题: {question}
    预期核心事实: {expected}
    实际回答: {actual}
    
    请仔细对比。如果实际回答意思正确或包含了预期核心事实，请严格只回复大写字母 "YES"，否则回复 "NO"。
    """
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.01  # 裁判的温度必须极低，保证每次评判标准一致
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except Exception as e:
        logging.error(f"调用裁判模型失败: {e}")
        return False

# ============ 4. 核心执行流程 ============
def run_tests():
    logging.info("🚀 开始 RAG 效果自动化测试 (LLM-as-a-Judge 模式)...")
    
    # 1. 初始化 RAG 系统的向量库
    chunks = load_and_split_document("cqupt.txt") # 确保这个 txt 文件在同级目录
    retriever = create_vector_db(chunks)
    
    # 2. 初始化裁判模型客户端
    load_dotenv()
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        logging.error("未找到 ZHIPU_API_KEY，请检查 .env 文件！")
        return
    client = ZhipuAI(api_key=api_key)
    
    correct = 0
    total = len(test_cases)
    
    for i, (q, expected) in enumerate(test_cases, 1):
        # 调用你的 RAG 大脑获取答案
        actual_ans, _ = ask_question(q, retriever)
        
        # 裁判模型进行智能比对
        is_correct = evaluate_with_llm(client, q, expected, actual_ans)
        
        if is_correct:
            logging.info(f"✅ [{i}/{total}] 通过 | 问题: {q}")
            correct += 1
        else:
            logging.warning(f"❌ [{i}/{total}] 翻车 | 问题: {q}")
            logging.warning(f"   🎯 预期事实: {expected}")
            logging.warning(f"   🤖 实际回答: {actual_ans}")

    logging.info(f"📊 测试完成！智能评估准确率: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    run_tests()