"""CQUPT RAG — FastAPI application entry point."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load env before importing modules that validate secrets at import time.
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Chroma
import uvicorn

from rag import init_rag_system, init_bm25_index, get_strategy
from routers import auth_router, chat_router, documents_router, interview_router, memories_router, quick_facts_router, sessions_router, settings_router
from services.web_search import build_tool_registry
from services.logging_config import configure_logging
from vector_store import get_embeddings
from database import Base, engine, ensure_database_exists

configure_logging(
    log_dir=Path(os.getenv("LOG_DIR", "logs")),
    level=os.getenv("LOG_LEVEL", "INFO"),
    backup_count=int(os.getenv("LOG_BACKUP_DAYS", "14")),
)

LONG_TERM_MEMORY_DIR = __import__("pathlib").Path("./chroma_memory_db")
LONG_TERM_MEMORY_COLLECTION = "chat_long_term_memory"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8015"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    def init_long_term_memory_store() -> Chroma:
        LONG_TERM_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(LONG_TERM_MEMORY_DIR),
            collection_name=LONG_TERM_MEMORY_COLLECTION,
            embedding_function=get_embeddings(),
        )

    import asyncio
    from database import ensure_interview_columns, ensure_session_overflow_columns

    await asyncio.to_thread(ensure_database_exists)
    await asyncio.to_thread(Base.metadata.create_all, bind=engine)
    await asyncio.to_thread(ensure_session_overflow_columns)
    await asyncio.to_thread(ensure_interview_columns)
    try:
        from services.interview import purge_expired_resume_texts

        purged = await asyncio.to_thread(purge_expired_resume_texts)
        if purged:
            logging.getLogger(__name__).info("purged expired resume_text on %s sessions", purged)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning("resume purge skipped", exc_info=True)
    retriever_data, long_term_store = await asyncio.gather(
        asyncio.to_thread(init_rag_system),
        asyncio.to_thread(init_long_term_memory_store),
    )
    retriever, retriever_init_info = retriever_data
    app.state.retriever = retriever
    app.state.retriever_init_info = retriever_init_info
    app.state.long_term_memory_store = long_term_store
    app.state.tool_registry = build_tool_registry()

    if get_strategy() == "hybrid":
        await asyncio.to_thread(init_bm25_index)

    yield


_raw_cors = os.getenv("CORS_ORIGINS", "")
if _raw_cors.strip():
    CORS_ORIGINS = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]
else:
    CORS_ORIGINS = ["http://localhost:5074", "http://127.0.0.1:5074"]

app = FastAPI(
    title="重邮极客 AI 大脑",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    from services.log_context import bind_log_context, reset_log_context

    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    session_match = re.match(r"^/sessions/([^/]+)", request.url.path)
    session_id = session_match.group(1) if session_match else ""
    tokens = bind_log_context(request_id=request_id, user_id="", session_id=session_id)
    request.state.request_id = request_id
    request.state.session_id = session_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logging.exception("Unhandled HTTP request failure")
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        access_tokens = bind_log_context(
            user_id=getattr(request.state, "user_id", ""),
            session_id=getattr(request.state, "session_id", ""),
        )
        try:
            logging.getLogger("cqupt_rag.access").info(
                "http_request method=%s path=%s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
        finally:
            reset_log_context(access_tokens)
        reset_log_context(tokens)

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(settings_router)
app.include_router(memories_router)
app.include_router(quick_facts_router)
app.include_router(interview_router)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    client_ip = request.client.host if request.client else "unknown"
    errors = exc.errors()
    logging.warning("Validation failed from ip=%s details=%s", client_ip, errors)

    details = [
        {
            "loc": " -> ".join(str(item) for item in error.get("loc", [])),
            "msg": error.get("msg", ""),
        }
        for error in errors
    ]

    return JSONResponse(
        status_code=422,
        content={
            "error_code": "RAG_422",
            "message": "参数校验失败，请检查输入规范。",
            "details": details,
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        app="api:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )
