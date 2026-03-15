"""
RAG核心模块 - 重邮知识库问答系统
"""
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# ============ LangChain 核心组件 ============
from langchain_community.document_loaders import TextLoader          # 文档加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_huggingface import HuggingFaceEmbeddings              # 向量化翻译官 (修复了最新版导入路径)
from langchain_community.vectorstores import Chroma                  # 向量数据库

# ============ 1. 初始化 ============
# 激活 .env 文件，加载 ZHIPU_API_KEY
load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

# ============ 2. 文档加载与切片 ============
def load_and_split_document(file_path="cqupt.txt"):
    """加载txt文档并智能切片"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到知识库文件：{file_path}，请先创建！")
        
    # 加载文档（指定utf-8避免中文乱码）
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    
    # 切片配置：优化为短切片，防止“切片连带效应”造成幻觉
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,      # 每块大约包含100个字符
        chunk_overlap=20,    # 保留20个字符的重叠，防止语义被生硬截断
        separators=["\n\n", "\n", "。", " ", ""]
    )
    return text_splitter.split_documents(docs)

# ============ 3. 向量数据库 ============
def create_vector_db(chunks, persist_dir="./chroma_db"):
    """将文本切片转化为向量并存入数据库"""
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",  # 中文专用Embedding模型
        model_kwargs={'device': 'cpu'},                 # 使用CPU计算
        encode_kwargs={'normalize_embeddings': True}    # 【关键防坑】强制归一化，为计算余弦相似度做准备
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}    # 【关键防坑】强制使用余弦相似度，避免L2距离算出版负数分数
    )
    
    # 返回检索器：取最相似的 Top 3，且相似度必须大于 0.3
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3}  
    )

# ============ 4. 问答核心大脑 ============
def ask_question(question, retriever):
    """RAG全流程：检索参考资料 + 大模型生成答案"""
    # Step1: 检索相关文档片段 (使用最新的 invoke 方法)
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step2: 构造Prompt（用严格的指令限制大模型胡编乱造）
    prompt = f"""你是一名重庆邮电大学助手，请严格根据以下资料回答问题。
要求：
1. 仅提取与用户问题直接相关的信息，不要回答资料中的无关内容。
2. 如果资料中没有相关信息，请直接回答“无法确定”，绝不能编造答案。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""
    
    # Step3: 调用智谱大模型
    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # 降低温度值，让模型的回答更严谨、更死板
    )
    return response.choices[0].message.content, context  # 返回“最终答案”和“检索到的原文”

# ============ 5. 本地终端测试入口 ============
if __name__ == "__main__":
    print("🚀 正在初始化RAG系统 (初次运行需下载Embedding模型，请耐心等待)...")
    chunks = load_and_split_document()
    retriever = create_vector_db(chunks)
    print("✅ 系统就绪！输入问题（输入'退出'结束）\n")
    
    while True:
        q = input("❓ 你的问题：").strip()
        if q.lower() in ["退出", "q", "quit"]:
            break
        if not q:
            continue
            
        # 调用核心问答函数
        answer, retrieved = ask_question(q, retriever)
        
        # 打印结果
        print(f"\n💡 回答：\n{answer}")
        
        # 友好展示检索片段（如果没有查到，就显示提示）
        preview = retrieved[:200] + "..." if retrieved else "未检索到相关片段，模型将根据 Prompt 规则拒绝回答"
        print(f"\n🔍 检索到的原文片段：\n{preview}")  
        print("\n" + "="*50 + "\n")