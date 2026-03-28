# Enterprise Campus Knowledge Base & Tool-Orchestrated AI Agent

一个面向校园场景的知识库问答与工具调度项目，围绕学生手册知识检索、多轮会话记忆、工具调用和 Web 交互构建。项目当前基于本地 PDF 知识库、FastAPI 后端、Streamlit 前端、MySQL 会话存储与 Chroma 向量检索实现，适合作为“企业级校园知识库与工具调度 AI Agent”的课程项目、作品集项目和后续工程化演进基础。

## 项目亮点

- 基于学生手册 PDF 自动构建本地向量知识库
- 支持校园制度问答、知识检索增强生成与来源片段注入
- 支持多轮会话上下文、长期记忆摘要沉淀与会话隔离
- 支持模型基于 Function Calling 的工具调度
- 内置天气查询、课表查询两个示例工具，便于后续替换成真实服务
- 提供 FastAPI 接口与 Streamlit 聊天前端
- 提供数据库初始化、示例数据写入、数据查询与异步评测脚本

## 适用场景

- 校园知识库问答
- 学生手册、规章制度、办事流程问答助手
- 工具调用型智能体课程设计
- RAG + 记忆 + 工具调度一体化 Demo

## 系统架构

```text
用户 / 前端
   |
   v
Streamlit Web UI / FastAPI API
   |
   v
Agent 编排层
   |- 短期记忆：MySQL 最近多轮消息
   |- 长期记忆：Chroma Memory 摘要检索
   |- 知识检索：学生手册 PDF -> Chroma Vector DB
   |- 工具调度：天气 / 课表 Function Calling
   |
   v
GLM-4 生成最终回答
```

## 技术栈

- LLM: `glm-4`
- Embedding: 智谱 `embedding-2`
- RAG: `LangChain + Chroma`
- API: `FastAPI + Uvicorn`
- Frontend: `Streamlit`
- Database: `MySQL + SQLAlchemy`
- Document Processing: `PyMuPDF` / `PyPDF`

## 核心能力

### 1. 校园知识库问答

项目从本地学生手册 PDF 读取内容、切块、建立向量索引，并在提问时检索相关片段后交给大模型生成答案。

### 2. 多层记忆机制

- 短期记忆：读取当前会话最近多轮消息
- 长期记忆：将高价值对话窗口摘要后写入 Chroma Memory
- 知识记忆：来自学生手册 PDF 的制度知识

三类上下文会在后端统一拼装，再发送给模型。

### 3. 工具调度 Agent

项目在聊天接口中实现了一轮工具调用闭环：

1. 先让模型判断是否需要调工具
2. 若模型返回 `tool_calls`，后端解析参数
3. 本地执行对应工具函数
4. 将工具结果作为 `tool` 消息回传模型
5. 模型生成最终答案

当前示例工具：

- `get_weather`
- `get_class_schedule`

### 4. 基础安全与工程化处理

- 对 `session_id` / `user_id` 做会话归属校验
- 对输入做基础 Prompt 注入关键词拦截
- 启动时自动初始化知识检索器与长期记忆向量库
- 支持索引元数据保存与本地持久化

## 项目结构

```text
.
├─ api.py               # FastAPI 主服务，Agent 编排、记忆组装、工具调度
├─ web.py               # Streamlit 聊天前端
├─ rag.py               # RAG 主流程：检索、重排、上下文构建、问答
├─ vector_store.py      # PDF 加载、切块、向量库构建与加载
├─ settings.py          # 模型、索引、切块与路径配置
├─ tools.py             # Function Calling 工具定义与本地实现
├─ database.py          # MySQL 连接与数据库初始化
├─ models.py            # SQLAlchemy ORM 模型
├─ main.py              # 初始化数据库表结构
├─ insert_data.py       # 写入示例会话数据
├─ query_data.py        # 查询示例会话数据
├─ evaluate.py          # 异步评测脚本
├─ async.py             # evaluate.py 的兼容入口
├─ test.py              # 本地轻量测试
├─ test_cases.json      # 评测样例
├─ chroma_db/           # 学生手册知识库向量索引（本地生成）
└─ chroma_memory_db/    # 长期记忆向量索引（本地生成，不建议入库）
```

## 环境要求

- Python 3.10+
- MySQL 8+
- 智谱 API Key

## 安装依赖

```bash
pip install fastapi uvicorn streamlit requests sqlalchemy pymysql PyMuPDF pypdf langchain langchain-community langchain-text-splitters chromadb python-dotenv zhipuai
```

## 环境变量配置

在项目根目录创建 `.env` 文件：

```env
ZHIPU_API_KEY=your_zhipu_api_key
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=cqupt_rag
```

说明：

- `ZHIPU_API_KEY` 用于大模型与向量 Embedding
- MySQL 变量用于会话消息和用户数据存储
- `.env` 已被 `.gitignore` 忽略，不会被提交到 GitHub

## 知识源配置

项目默认会从 [settings.py](/d:/ai/ai/settings.py) 中的 `PDF_PATH` 读取学生手册 PDF。  
如果你更换知识文档路径，修改这里即可：

```python
PDF_PATH = Path(r"C:\path\to\your\manual.pdf")
```

## 快速开始

### 1. 初始化数据库

```bash
python main.py
```

### 2. 启动后端服务

```bash
python api.py
```

默认监听：

```text
http://127.0.0.1:8000
```

### 3. 启动 Web 前端

```bash
streamlit run web.py
```

### 4. 访问聊天页面

启动后可以在浏览器中与 Agent 进行对话，测试：

- 校园制度问答
- 天气查询
- 课表查询
- 多轮上下文与记忆效果

## API 使用方式

### 健康检查

```bash
curl http://127.0.0.1:8000/
```

### 聊天接口

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"user_10001\",\"session_id\":\"session_10001\",\"new_message\":\"休学申请一般需要满足什么条件？\"}"
```

返回示例：

```json
{
  "reply": "..."
}
```

## 常用脚本

### 写入示例数据

```bash
python insert_data.py
```

### 查询示例数据

```bash
python query_data.py
```

### 异步评测

```bash
python evaluate.py
```

或：

```bash
python async.py
```

### 轻量测试

```bash
python test.py
```

## RAG 检索流程

当前项目的知识检索流程是：

1. 读取学生手册 PDF
2. 使用 `RecursiveCharacterTextSplitter` 切块
3. 使用智谱 `embedding-2` 生成向量
4. 写入本地 `Chroma`
5. 提问时先做向量召回
6. 再做一层轻量关键词重排
7. 将命中片段拼入 Prompt 交给 `glm-4`

## 记忆机制说明

### 短期记忆

从 MySQL 中读取当前会话最近多轮消息，作为即时上下文。

### 长期记忆

当对话窗口达到一定条件后，系统会把高价值对话压缩成摘要，写入 `chroma_memory_db/`，后续按当前问题做相似度召回。

### 会话隔离

长期记忆检索时会按 `session_id` 做过滤，避免跨会话串线。

## 工具调度说明

当前仓库内置的是示例工具实现，适合演示 Function Calling 闭环。后续如果要升级为真实企业级校园智能体，可以将这些函数替换为：

- 教务系统课表接口
- 校园天气或城市天气接口
- 办事大厅流程查询接口
- 校园通知公告接口
- 学校知识中台 / 企业内部知识库接口

## GitHub 与隐私说明

以下内容已配置为不上传：

- `.env`
- `chroma_db/`
- `chroma_memory_db/`
- `manual_index_meta.json`
- `__pycache__/`

因此本地密钥、索引缓存和运行时文件不会被提交。

## 后续可扩展方向

- 接入真实校园业务工具
- 引入更强的重排模型
- 增加用户身份体系与权限控制
- 接入对象存储与多文档知识库
- 增加日志、监控、Tracing 与评测看板
- 支持 Docker 化部署

## 项目定位

这是一个以校园场景为核心、具备知识检索、记忆管理与工具调度能力的 Agent 项目原型。  
它已经具备“企业级校园知识库与工具调度 AI Agent”的基础骨架，适合继续向更完整的工程化版本演进。
