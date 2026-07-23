# CQUPT-RAG-Assistant

重邮校园智能助手：面向学生手册 / 制度文档的 **RAG 问答**，外加面向求职的 **面试题库助手**。

- 校园问答：多策略混合检索、证据门控、流式对话、可确认画像记忆、PDF 原页溯源
- 面试助手：JD + 简历出题、联网搜面经、选择题作答 / 错题报告 / 薄弱点重出、简答题口语答案与题目讲解

## 端口约定（本项目固定）

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://127.0.0.1:5074 | Vite dev，`strictPort`，代理 `/api` → 后端 |
| 后端 | http://127.0.0.1:8015 | FastAPI / WebSocket |
| API 文档 | http://127.0.0.1:8015/docs | 可选 |

> 刻意避开常见默认端口 `5173` / `8000`，避免与本机其他项目冲突。

## Tech Stack

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek（默认 `deepseek-v4-flash`） |
| Embedding | 智谱 `embedding-2`（仅向量化，不走对话） |
| 后端 | Python / FastAPI / LangChain |
| 向量库 | ChromaDB + 父子文章切块索引 |
| 混合检索 | 向量 + BM25（jieba / rank_bm25）+ RRF 融合 |
| 联网搜索 | Tavily（可选；面试面经 / 聊天工具） |
| 数据库 | MySQL + SQLAlchemy ORM |
| 认证 | JWT（HttpOnly Cookie / Bearer）+ bcrypt |
| 前端 | React 19 / TypeScript / Zustand / Tailwind / Vite |
| 实时通信 | WebSocket 流式输出 |

## 产品能力

### 1. 校园 RAG 问答 (`/chat`)

```text
用户提问
  │
  ├─ 指代消解 / 跟进问题改写（可选 LLM）
  ├─ Quick Facts 快答（高频事实短路）
  ├─ 意图分流：闲聊直接回 / 知识问答进 RAG / 工具调用
  │
  ├─ 检索链路（默认 hybrid）
  │   ├─ 查询归一化 + 领域别名
  │   ├─ 向量召回 + BM25
  │   ├─ RRF 融合 + 父子 chunk 回填
  │   ├─ 证据门控（分数 / 覆盖率 / 动态信息）
  │   └─ 弱证据时按 REWRITE_MODE 触发查询改写
  │
  ├─ 可选联网：Tavily / MCP 工具（配置开启后）
  │
  ├─ 上下文
  │   ├─ 近窗：最近 N 轮原文（MySQL）
  │   ├─ 超窗：单份滚动摘要（sessions 表字段）
  │   └─ 画像：用户确认后的偏好/档案（可删，不静默记全聊天）
  │
  └─ 流式回答 + 来源引用 + PDF 原页 / 缩略图预览
```

### 2. 面试题库助手 (`/interview`)

```text
公司 + 岗位 + JD + 简历(PDF/DOCX/粘贴)
  → 可选 Tavily 搜面经参考
  → 选择题 20 道（选项去重 + LLM 复核质检）
  → 简答题 30 道（分批生成防截断；5 类题型）
  → 前端独立测验：作答 → 提交出分 → 解析
  → 薄弱点报告（Markdown）+ 针对错题生成第 2 套
  → 口语答案 150–220 字 +「题目讲解」（教你怎么理解题）
  → 导出 Markdown / 历史题库回看删除
```

## Key Features

**检索与生成**
- 可切换检索策略：baseline / 关键词重排 / 查询改写 / hybrid（BM25+向量 RRF）
- 父子文章切块、证据门控、gated rewrite、引用页扩展
- WebSocket 流式输出、思考过程与来源引用

**记忆与个性化**
- 工作记忆：近窗原文 + 超窗滚动摘要（MySQL，非向量库）
- 画像记忆：正则 + LLM 抽取 → 用户确认后落库，可管理删除
- 会话全文检索、重命名、多会话管理

**知识库体验**
- 制度文档列表浏览、PDF 原页 PNG / 缩略图按需渲染（PyMuPDF）
- Quick Facts 高频事实卡

**面试助手**
- 面经联网增强、选择题测验、错题报告、薄弱点重出
- 简答题分类、口语答案、题目讲解、等待动画（分阶段进度）

**工程**
- 手机号注册登录、JWT、模型设置、中英文 i18n
- 检索评测脚本与 metrics（`evaluation/`）

## Quick Start

### 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

需要本机 MySQL，并准备 DeepSeek / 智谱 API Key。

### 2. 配置环境变量

```bash
cp .env.example .env
```

必填：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | 对话 / 改写 / 摘要 / 出题 |
| `ZHIPU_API_KEY` | embedding-2 |
| `JWT_SECRET_KEY` | ≥32 字符随机串 |
| `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_HOST` / `MYSQL_DATABASE` | 数据库 |

常用可选：

| 变量 | 默认 | 说明 |
|------|------|------|
| `API_PORT` | `8015` | 后端端口 |
| `CORS_ORIGINS` | `http://localhost:5074,...` | 前端源 |
| `TAVILY_API_KEY` | 空 | 面经搜索 / 联网；不填则自动降级 |
| `WEB_SEARCH_ENABLED` | `false` | 聊天侧联网开关 |
| `REWRITE_MODE` | `auto` | `auto` / `on` / `off` |
| `LLM_MODEL` | `deepseek-v4-flash` | 主模型 |

完整注释见 [`.env.example`](.env.example)。

### 3. 初始化数据库

```bash
python main.py
```

启动 API 时也会自动 `create_all` 并补齐面试 / 超窗摘要等列迁移。

### 4. 启动服务

```bash
# 终端 1 — 后端 :8015
python api.py

# 终端 2 — 前端 :5074
cd frontend && npm run dev
```

打开 http://127.0.0.1:5074

- 登录 / 注册 → 校园问答 `/chat`
- 侧边栏进入面试助手 `/interview`

### 5. 测试

```bash
# 后端
./.venv/bin/python -m pytest tests/ -q

# 前端
cd frontend && npm test -- --run
```

## Project Structure

```text
├── api.py                 # FastAPI 入口、CORS、生命周期、路由挂载
├── main.py                # DB 初始化辅助
├── rag.py                 # RAG 主流程与策略分发
├── vector_store.py        # 文档向量化 / Chroma
├── settings.py            # 配置中心
├── models.py              # ORM：User / Session / Message / Memory / Interview...
├── database.py            # 引擎、列级 ensure 迁移
├── security.py            # JWT + 密码哈希
├── tools.py               # Function Calling 工具定义
├── quick_facts.json       # 高频事实
├── evaluate.py / experiment.py
├── evaluation/            # 检索评测 runner + metrics
├── routers/
│   ├── auth.py            # 注册 / 登录 / refresh
│   ├── chat.py            # REST + WebSocket 聊天
│   ├── sessions.py        # 会话 CRUD / 搜索 / 重命名
│   ├── documents.py       # 知识库列表 + PDF 页渲染
│   ├── memories.py        # 画像记忆确认 / 管理
│   ├── interview.py       # 面试题库生成 / 报告 / 重出
│   ├── quick_facts.py     # 快答事实
│   └── settings.py        # 前端可读的模型等设置
├── services/
│   ├── llm.py / rewriter.py / hybrid.py / retrieval.py
│   ├── parent_child_index.py / article_chunker.py / evidence.py
│   ├── confidence.py / query_normalize.py
│   ├── working_memory.py / session_summary.py / conversation_context.py
│   ├── memory_manager.py / memory_candidates.py
│   ├── interview.py       # 出题、质检、报告、分批 QA
│   ├── web_search.py / mcp_tools.py / tool_registry.py
│   ├── quick_facts.py
│   └── logging_config.py / log_context.py
├── tests/                 # pytest
└── frontend/
    └── src/
        ├── api/client.ts
        ├── pages/         # LoginPage / ChatPage / InterviewPage
        ├── components/
        │   ├── chat/      # 消息、侧栏、引用、记忆、设置...
        │   ├── interview/ # 导出 Markdown、生成等待动画
        │   ├── auth/ · ui/
        ├── i18n/          # zh / en
        └── store/         # auth / chat / theme / toast
```

## API 一览

| 分组 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 认证 | POST | `/auth/register` `/auth/login` `/auth/refresh` | 注册登录刷新 |
| 聊天 | POST | `/chat` | 同步聊天 |
| 聊天 | WS | `/ws/chat` | 流式聊天 |
| 会话 | GET/PATCH/DELETE | `/sessions` `/sessions/{id}` | 列表 / 重命名 / 删除 |
| 会话 | GET | `/sessions/{id}/messages` `/sessions/search` | 消息 / 全文搜 |
| 文档 | GET | `/documents` `/documents/{id}/page/{n}` | 列表 / 原页 PNG |
| 记忆 | GET/POST/DELETE | `/memories` … | 画像列表 / 确认 / 删除 |
| 快答 | GET | `/quick-facts` | 高频事实 |
| 设置 | GET | `/settings/...` | 模型等前端配置 |
| 面试 | POST | `/interview/generate` | 生成题库（较慢，约 2–3 分钟） |
| 面试 | GET/DELETE | `/interview/sessions` `/interview/sessions/{id}` | 历史题库 |
| 面试 | POST | `/interview/sessions/{id}/report` | 薄弱点报告 |
| 面试 | POST | `/interview/sessions/{id}/regenerate-mcq` | 针对错题重出一套 |

> 浏览器经 Vite 访问时走 `/api/*` 代理（去前缀后转发到 8015）。

## Evaluation

早期在小规模手册索引上做过策略对比（baseline → rewrite 等），整体准确率有过 41.7% → 66.7% 的提升记录。  
当前默认路径是 **hybrid + 证据门控 + 条件改写 + 父子索引**；更新的评测记录与实验笔记在 `docs/evaluation/` 与 `evaluation/`。

```bash
# 示例：跑检索评测（需本地索引与用例就绪）
./.venv/bin/python -m evaluation.run_retrieval
```

## License

MIT
