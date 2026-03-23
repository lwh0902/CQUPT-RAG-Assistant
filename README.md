# CQUPT-RAG-Assistant

基于重邮《学生手册（教育管理篇）2025版》PDF 构建的 RAG 问答项目。

项目当前不再使用简单的 `txt` 知识库，而是固定绑定学生手册 PDF：
- 首次运行时解析 PDF、切片并创建向量库
- 后续运行时如果 PDF 未变化，则直接加载已有索引
- 问答时会返回答案，并展示检索到的原始片段和参考页码

## 项目特点

- 基于 `PyMuPDF` 读取学生手册 PDF
- 基于 `LangChain + Chroma` 构建本地向量检索
- 基于 `GLM-4` 生成最终回答
- 支持“首次构建，后续加载”的索引复用
- 支持 `asyncio` 并发评测，并发上限固定为 `3`

## 当前知识来源

项目当前唯一知识来源为本地 PDF：

`学生手册（教育管理篇）2025版.pdf`

默认路径配置在 [`settings.py`](/d:/ai/ai/settings.py) 中：

```python
PDF_PATH = Path(r"C:\Users\皓\Downloads\学生手册（教育管理篇）2025版.pdf")
```

如果你后续更换 PDF 存放位置，只需要修改这里。

## 主要文件说明

- [`app.py`](/d:/ai/ai/app.py)：Streamlit 页面入口
- [`rag.py`](/d:/ai/ai/rag.py)：RAG 主流程，负责初始化、检索、问答和异步包装
- [`pdf_loader.py`](/d:/ai/ai/pdf_loader.py)：PDF 读取与切片
- [`vector_store.py`](/d:/ai/ai/vector_store.py)：向量库创建、加载和索引元数据管理
- [`settings.py`](/d:/ai/ai/settings.py)：统一配置项
- [`evaluate.py`](/d:/ai/ai/evaluate.py)：异步并发评测主模块
- [`async.py`](/d:/ai/ai/async.py)：兼容入口，直接复用异步评测
- [`test.py`](/d:/ai/ai/test.py)：轻量本地冒烟测试
- [`test_cases.json`](/d:/ai/ai/test_cases.json)：学生手册评测题集

## 运行前准备

1. 安装依赖

```bash
pip install streamlit PyMuPDF langchain langchain-community langchain-huggingface chromadb sentence-transformers python-dotenv zhipuai
```

2. 配置环境变量

在项目根目录创建 `.env` 文件，并写入：

```env
ZHIPU_API_KEY=你的智谱API密钥
```

## 启动方式

启动 Web 问答界面：

```bash
streamlit run app.py
```

运行异步评测：

```bash
python evaluate.py
```

运行轻量冒烟测试：

```bash
python test.py
```

## 索引机制说明

项目会自动在本地生成：

- `chroma_db/`：向量库目录
- `manual_index_meta.json`：索引元数据文件

启动时会自动判断：
- 如果学生手册 PDF 没变，则直接加载已有向量库
- 如果 PDF 发生变化，则重新构建索引

## 适用场景

这个项目更适合：
- 课程设计
- RAG 项目练习
- 基于固定制度文件的智能问答

当前版本特别适合问答：
- 奖学金
- 学籍管理
- 转学、休学、退学
- 纪律处分
- 学生申诉
- 学业与毕业要求

## 注意事项

- 首次运行如果本地没有 Embedding 模型缓存，可能需要联网下载
- `.env` 不应上传到 GitHub
- 如果更换了学生手册 PDF，系统会自动重新建索引
