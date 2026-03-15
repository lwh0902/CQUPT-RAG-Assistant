"""
自动化测试脚本 - 工业级 RAG 效果评估 (Asyncio 异步高并发版)
文件名: evaluate_rag_async.py
"""
import os
import time
import asyncio
import logging
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 导入你的同步 RAG 函数
from rag import load_and_split_document, create_vector_db, ask_question
from evaluate import test_cases, evaluate_with_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 【面试高频考点】并发控制：设置信号量（Semaphore）
# 为什么要有这一步？因为如果你瞬间向 API 发起 1000 个请求，会被直接封 IP 或触发 Rate Limit（限流）。
# 这里限制最多同时处理 5 个并发请求，保护我们的 API 不崩。
MAX_CONCURRENCY = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

async def process_single_case(client: ZhipuAI, retriever, idx: int, q: str, expected: str) -> bool:
    """
    处理单个测试用例的异步任务
    """
    async with semaphore:  # 获取信号量，控制并发数
        try:
            # 【核心底层原理】由于你的 ask_question 和 evaluate_with_llm 里面使用的是同步的 API 调用（会阻塞）
            # 我们必须使用 asyncio.to_thread 将同步的、会阻塞的代码扔到后台线程池去执行，
            # 这样就不会阻塞主事件循环（Event Loop），实现真正的并发！
            
            # 1. 异步获取 RAG 回答
            actual_ans, _ = await asyncio.to_thread(ask_question, q, retriever)
            
            # 2. 异步进行 LLM 裁判打分
            is_correct = await asyncio.to_thread(evaluate_with_llm, client, q, expected, actual_ans)
            
            status = "✅" if is_correct else "❌"
            logging.info(f"{status} [{idx}] 评测完毕 | 问题: {q}")
            
            if not is_correct:
                logging.warning(f"   🎯 预期: {expected} | 🤖 实际: {actual_ans}")
                
            return is_correct
        except Exception as e:
            logging.error(f"❌ [{idx}] 处理异常 | 问题: {q} | 报错: {e}")
            return False

async def main():
    logging.info("🚀 启动高并发 RAG 评测系统...")
    start_time = time.time()
    
    # 1. 初始化（本地知识库处理较快，保持同步即可）
    chunks = load_and_split_document("cqupt.txt")
    retriever = create_vector_db(chunks)
    
    load_dotenv()
    client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
    
    # 2. 构建异步任务列表
    tasks = []
    for i, (q, expected) in enumerate(test_cases, 1):
        # 创建任务但不立即执行，统统塞进任务列表
        task = asyncio.create_task(process_single_case(client, retriever, i, q, expected))
        tasks.append(task)
        
    # 3. 【一键并发】使用 asyncio.gather 同时启动所有任务，并等待它们全部完成
    results = await asyncio.gather(*tasks)
    
    # 4. 统计结果
    correct_count = sum(results) # results 是一个包含所有 is_correct (True/False) 的列表
    total_count = len(test_cases)
    
    end_time = time.time()
    cost_time = end_time - start_time
    
    logging.info("=" * 40)
    logging.info(f"📊 并发测试完成！")
    logging.info(f"🎯 准确率: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    logging.info(f"⏱️ 总体耗时: {cost_time:.2f} 秒")
    logging.info("=" * 40)

if __name__ == "__main__":
    # 运行 asyncio 的标准起手式
    asyncio.run(main())