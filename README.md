# CQUPT-RAG-Assistant

基于 RAG 与 Function Calling 的校园智能问答系统，面向学生手册制度查询场景，支持多策略检索、Agent 工具调用、流式对话与长期记忆。

## Tech Stack

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek-V4-Flash |
| 后端 | Python / FastAPI / LangChain |
| 向量库 | ChromaDB + 智谱 embedding-2 |
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
  ├─ 上下文 / 记忆
  │   ├─ 近窗: MySQL 最近 N 轮原文（默认 6 轮）进 messages
  │   ├─ 超窗: 更早轮次 LLM 滚动摘要 1 份 → MySQL sessions 字段进 system
  │   ├─ 画像: 用户明确偏好/档案（少而准，可删）
  │   └─ 可选: 历史 Chroma 会话摘要检索（与超窗工作记忆分离）
  │
  └─ 安全防御: Prompt 注入检测 + 来源引用溯源
```

## Key Features

- **4 种可切换检索策略**: baseline → 关键词重排 → 查询改写 → BM25+向量混合 RRF 融合
- **Function Calling Agent**: 双轮工具调用链路，支持天气查询、课表查询等外部工具
- **工作记忆**: 近窗原文 + 超窗单份滚动摘要（MySQL）；画像仅保存明确表达且可删除，不自动记全聊天
- **LLM-as-Judge 评测框架**: 覆盖 fact / rule / multi_condition / refusal 四类问题
- **WebSocket 流式输出**: 前端实时 token 流 + 思考链动画 + 来源引用展示
- **完整用户系统**: 手机号注册 / 登录 / JWT 鉴权 / 会话管理
- **PDF 原页预览**: 答案引用支持点击跳转，PyMuPDF 按需渲染页码为 PNG，悬停显示缩略图、点击打开原页 + 原文片段
- **知识库资料浏览**: 弹窗列出系统挂载的全部制度文档（页数 / 大小 / 类型），支持翻页通读
- **会话全文检索**: SQL `ilike` 同时匹配会话标题与历史消息，返回高亮 snippet
- **会话重命名**: 双击侧边栏会话进入 inline 编辑，回车 / 失焦保存，ESC 取消
- **Spline 3D 登录页**: 黑底 + Spotlight 光晕 + Spline 机器人，移动端自适应

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
# 编辑 .env 填入 DEEPSEEK_API_KEY、ZHIPU_API_KEY（仅 embedding）和 MySQL 连接信息
```

### 3. 初始化数据库

```bash
python main.py
```

### 4. 启动服务

```bash
# 后端 (默认 :8015)
python api.py

# 前端 (默认 :5074，自动代理到后端)
cd frontend && npm run dev
```

访问 http://localhost:5074 即可使用。

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
│   ├── sessions.py         # 会话 CRUD + 全文搜索 + 重命名
│   └── documents.py        # PDF 原页 / 缩略图按需渲染 (PyMuPDF)
├── services/
│   ├── llm.py              # DeepSeek 客户端，流式输出
│   ├── rewriter.py         # LLM 查询改写 (1 → 2-4 子查询)
│   └── hybrid.py           # BM25 索引 + RRF 融合检索
├── data/
│   └── policy_graph.json   # 奖学金政策知识图谱
└── frontend/
    └── src/
        ├── api/client.ts   # Axios + WebSocket 客户端
        ├── pages/          # ChatPage, LoginPage (Spline 3D)
        ├── components/
        │   ├── chat/       # ChatMessage, Sidebar, SourceCitation, ThinkingChain,
        │   │               # PdfPreviewModal, KnowledgeBaseModal
        │   └── ui/         # Modal, Card, Spotlight, SplineScene, NeonButton
        ├── hooks/          # useHoverPreview (悬浮缩略图防抖)
        └── store/          # Zustand (auth, chat, theme, toast)
```

## API Endpoints

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` | 手机号注册 |
| POST | `/auth/login` | 登录，返回 JWT |
| POST | `/auth/check-phone` | 检查手机号是否已注册 |
| POST | `/chat` | 同步聊天 (REST) |
| WS | `/ws/chat` | 流式聊天 (WebSocket) |
| GET | `/sessions` | 会话列表 (分页) |
| GET | `/sessions/{id}/messages` | 会话消息 |
| PATCH | `/sessions/{id}` | 重命名会话 |
| DELETE | `/sessions/{id}` | 删除会话 |
| GET | `/sessions/search?q=` | 全文检索会话标题与历史消息 |
| GET | `/documents` | 知识库资料列表 |
| GET | `/documents/{id}/page/{n}` | 渲染 PDF 第 n 页为 PNG |
| GET | `/documents/{id}/page/{n}/thumbnail` | PDF 第 n 页缩略图 (0.4x) |

## License

MIT
