"""自动化测试脚本 - 评估RAG效果"""
# 【修复1】修改了包名导入格式，请确保你的源文件名为 rag_xiaoyiyuan.py
from rag import load_and_split_document, create_vector_db, ask_question

# 测试用例（10个文档内问题 + 10个文档外问题）
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

def run_tests():
    print("🧪 开始RAG效果测试...\n")
    chunks = load_and_split_document()
    retriever = create_vector_db(chunks)
    
    correct = 0
    for i, (q, expected) in enumerate(test_cases, 1):
        # 注意：这里假设你的 ask_question 函数返回两个值，例如 (回答文本, 检索到的文档源)
        ans, _ = ask_question(q, retriever)
        
        # 【修复2】去掉了 or "无法确定" in ans，防止测试结果产生虚高的假阳性
        is_correct = expected in ans
        
        status = "✅" if is_correct else "❌"
        # 增加容错：如果 ans 不是字符串，转为字符串防止切片报错
        ans_str = str(ans) 
        print(f"{status} [{i}/20] 问题：{q[:15]}... | 回答：{ans_str[:30]}...")
        if is_correct: 
            correct += 1
        else:  # <====== 【新加的代碼從這裡開始】意思是「否則（如果答錯了）」
            # 用 f-string 把我們提取出來的 expected (標準答案) 塞進句子裡打印出來
            print(f"    ⚠️ 這道題大模型翻車了，標準答案應該是：{expected}")
    
    print(f"\n📊 测试完成！准确率：{correct}/20 ({correct/20*100:.1f}%)")
    
    # 保存报告
    with open("test_report.md", "w", encoding="utf-8") as f:
        f.write(f"# RAG测试报告\n准确率：{correct}/20 ({correct/20*100:.1f}%)\n\n")
        f.write("## 测试用例明细（略）\n")
    print("📄 测试报告已保存至 test_report.md")

if __name__ == "__main__":
    run_tests()