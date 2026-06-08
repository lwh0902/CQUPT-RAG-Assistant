# CQUPT-RAG-Assistant

基于 RAG 与 Function Calling 的校园智能问答系统，面向学生手册制度查询场景，支持多策略检索、Agent 工具调用、流式对话与长期记忆。

## Tech Stack

| 层级 | 技术 |
|------|------|
| LLM | GLM-4.7-Flash (智谱 AI) |
| 后端 | Python / FastAPI / LangChain |
| 向量库 | ChromaDB + ZhipuAI Embedding |
| 混合检索 | rank_bm25 + jieba (BM25) / RRF 融合 |
| 数据库 | MySQL (SQLAlchemy ORM) |
| 认证 | JWT (bcrypt + HS256) |
| 前端 | React 19 / TypeScript / Zustand / Tailwind CSS |
| 实时通信 | WebSocket 流式输出 |

## Architecture

```text
用户提问
  │
  ├─ 意图分类 (LLM) ── 闲聊 → 直接回复
  │                  └─ 知识查询 → 进入 RAG
  │
  ├─ RAG 检索链路
  │   ├─ 查询改写 (1 → 2-4 子查询)
  │   ├─ 向量召回 + BM25 关键词检索
  │   ├─ RRF 融合排序
  │   └─ 领域术语精准重排 (30+ 术语词表)
  │
  ├─ Agent Loop (Function Calling)
  │   ├─ 第一轮: LLM 判断是否调用工具
  │   ├─ 工具执行 (天气 / 课表)
  │   └─ 第二轮: 注入工具结果，生成最终回答
  │
  ├─ 记忆系统
  │   ├─ 短期: MySQL 最近 8 条消息
  │   └─ 长期: LLM 摘要 → Chroma 向量检索
  │
  └─ 安全防御: Prompt 注入检测 + 来源引用溯源
```

## Key Features

- **4 种可切换检索策略**: baseline → 关键词重排 → 查询改写 → BM25+向量混合 RRF 融合
- **Function Calling Agent**: 双轮工具调用链路，支持天气查询、课表查询等外部工具
- **长短期双路记忆**: MySQL 短期对话 + Chroma 向量长期摘要，跨会话上下文保持
- **LLM-as-Judge 评测框架**: 覆盖 fact / rule / multi_condition / refusal 四类问题
- **WebSocket 流式输出**: 前端实时 token 流 + 思考链动画 + 来源引用展示
- **完整用户系统**: 手机号注册 / 登录 / JWT 鉴权 / 会话管理

## Evaluation Results

基于 4 份校园制度文档 (176 页 / 315 chunks) 构建 12 条结构化测试用例：

| 策略 | 整体准确率 | fact | rule | multi_condition | refusal |
|------|-----------|------|------|-----------------|---------|
| baseline | 41.7% | 60% | 0% | 50% | 100% |
| + 关键词重排 | 58.3% | **100%** | 25% | 50% | 0% |
| + 查询改写 | **66.7%** | **100%** | 25% | 50% | **100%** |
| + BM25+RRF 混合 | 41.7% | 60% | 0% | 50% | **100%** |

优化后整体准确率 41.7% → 66.7%，fact 类准确率 60% → 100%。

## Quick Start

### 1. 环境准备

```bash
# 后端依赖
pip install -r requirements.txt

# 前端依赖
cd frontend && npm install
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 ZHIPU_API_KEY 和 MySQL 连接信息
```

### 3. 初始化数据库

```bash
python main.py
```

### 4. 启动服务

```bash
# 后端 (默认 :8001)
python api.py

# 前端 (默认 :5173，自动代理到后端)
cd frontend && npm run dev
```

访问 http://localhost:5173 即可使用。

## Project Structure

```text
├── api.py                  # FastAPI 入口，CORS，路由挂载
├── rag.py                  # RAG 主流程 (4 种检索策略)
├── vector_store.py         # PDF/DOCX 向量化与 Chroma 索引
├── settings.py             # 项目配置中心
├── models.py               # SQLAlchemy ORM (User / Session / Message)
├── database.py             # MySQL 连接
├── security.py             # JWT 认证 + bcrypt 密码哈希
├── tools.py                # Function Calling 工具定义
├── evaluate.py             # LLM-as-Judge 评测框架
├── experiment.py           # 多策略对比实验
├── test_cases.json         # 结构化测试用例
├── routers/
│   ├── auth.py             # 手机号注册 / 登录 / 鉴权
│   ├── chat.py             # REST + WebSocket 聊天，Agent Loop，记忆管理
│   └── sessions.py         # 会话 CRUD
├── services/
│   ├── llm.py              # ZhipuAI 客户端，流式输出
│   ├── rewriter.py         # LLM 查询改写 (1 → 2-4 子查询)
│   └── hybrid.py           # BM25 索引 + RRF 融合检索
├── data/
│   └── policy_graph.json   # 奖学金政策知识图谱
└── frontend/
    └── src/
        ├── api/client.ts   # Axios + WebSocket 客户端
        ├── pages/          # ChatPage, LoginPage
        ├── components/     # ChatMessage, Sidebar, SourceCitation, ThinkingChain
        └── store/          # Zustand (auth, chat, theme, toast)
```

## API Endpoints

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` | 手机号注册 |
| POST | `/auth/login` | 登录，返回 JWT |
| POST | `/chat` | 同步聊天 (REST) |
| WS | `/ws/chat` | 流式聊天 (WebSocket) |
| GET | `/sessions` | 会话列表 (分页) |
| GET | `/sessions/{id}/messages` | 会话消息 |
| DELETE | `/sessions/{id}` | 删除会话 |

## License

MIT
