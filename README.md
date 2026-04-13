# Campus Knowledge QA System

一个面向课程实验的校园知识库智能问答系统，包含：

- `FastAPI` 后端接口
- `Streamlit` Web 前端
- `MySQL` 会话与消息持久化
- `Chroma` 向量库与长期记忆
- `GLM-4` 问答与工具调用

这套项目现在已经具备两个更适合实验报告描述的核心模块：

1. 校园知识库智能问答
2. 历史会话管理与记忆检索

天气和课表工具查询可以作为扩展功能展示。

## Project Structure

```text
.
├── api.py              # FastAPI 后端，负责聊天、工具调用、历史会话接口
├── web.py              # Streamlit 前端，支持聊天和历史会话切换
├── rag.py              # RAG 主流程
├── vector_store.py     # PDF 向量化与 Chroma 索引
├── database.py         # MySQL 连接与建库
├── models.py           # SQLAlchemy ORM 模型
├── main.py             # 初始化数据库表
├── evaluate.py         # 异步评测脚本
├── settings.py         # 项目配置
├── requirements.txt    # Python 依赖
└── .env.example        # 环境变量示例
```

## Core Features

### 1. 校园知识库智能问答

- 从学生手册 PDF 构建向量索引
- 使用 RAG 检索相关片段
- 基于 `glm-4` 生成回答

### 2. 历史会话管理

- 用户消息和助手消息持久化到 MySQL
- 前端侧边栏可查看历史会话
- 支持切换旧会话继续提问
- 会话标题根据首条问题自动生成

### 3. 记忆与工具扩展

- 短期记忆：读取当前会话最近多轮消息
- 长期记忆：对高价值对话做摘要并写入 Chroma
- 工具调用：天气、课表示例工具

## Environment

复制 `.env.example` 为 `.env`，然后按需修改：

```env
ZHIPU_API_KEY=your_zhipu_api_key
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=cqupt_rag
PDF_PATH=student_manual.pdf
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
API_BASE_URL=http://127.0.0.1:8000
```

说明：

- `PDF_PATH` 只在需要重建知识库索引时使用
- `API_BASE_URL` 供 `web.py` 连接后端接口
- `WS_BASE_URL` 可选；未设置时前端会根据 `API_BASE_URL` 自动推导

## Install

```bash
pip install -r requirements.txt
```

## Run Locally

1. 初始化数据库表

```bash
python main.py
```

2. 启动后端

```bash
python api.py
```

3. 启动前端

```bash
streamlit run web.py
```

4. 浏览器访问

```text
http://localhost:8501
```

## API

### 健康检查

```bash
curl http://127.0.0.1:8000/
```

### 聊天接口

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"user_10001\",\"session_id\":\"session_10001\",\"new_message\":\"休学一般需要满足什么条件？\"}"
```

### 会话列表

```bash
curl "http://127.0.0.1:8000/sessions?user_id=user_10001"
```

### 历史消息

```bash
curl "http://127.0.0.1:8000/sessions/session_10001/messages?user_id=user_10001"
```

## Deployment Notes

如果是课程实验，推荐两种最省事的部署方式：

1. `ngrok + 本地运行`
   适合快速完成“公网可访问 URL”要求。
2. `Render / Railway`
   适合长期保留演示地址，但需要把 MySQL、环境变量和 PDF/索引准备好。

部署前建议确认：

- `API_BASE_URL` 已改成公网后端地址
- `PDF_PATH` 已配置为服务器可访问路径，或已提前生成向量库
- `MYSQL_*` 环境变量可在目标环境正常连接

## Experiment Mapping

这个项目可以直接对应实验要求中的系统实现部分：

- 前端交互：`web.py`
- 后端逻辑：`api.py`
- 数据持久化：`database.py` + `models.py`
- 两个核心业务功能：
  - 校园知识库问答
  - 历史会话管理

如果要写实验报告，建议系统名称使用：

`校园知识库智能问答系统`
