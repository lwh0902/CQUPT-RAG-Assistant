"""CQUPT RAG — FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Chroma
import uvicorn

from rag import init_rag_system, init_bm25_index, get_strategy
from routers import auth_router, chat_router, documents_router, sessions_router
from vector_store import get_embeddings

load_dotenv()

LONG_TERM_MEMORY_DIR = __import__("pathlib").Path("./chroma_memory_db")
LONG_TERM_MEMORY_COLLECTION = "chat_long_term_memory"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
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
    retriever_data, long_term_store = await asyncio.gather(
        asyncio.to_thread(init_rag_system),
        asyncio.to_thread(init_long_term_memory_store),
    )
    retriever, retriever_init_info = retriever_data
    app.state.retriever = retriever
    app.state.retriever_init_info = retriever_init_info
    app.state.long_term_memory_store = long_term_store

    if get_strategy() == "hybrid":
        await asyncio.to_thread(init_bm25_index)

    yield


_raw_cors = os.getenv("CORS_ORIGINS", "")
if _raw_cors.strip():
    CORS_ORIGINS = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]
else:
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

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

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(chat_router)
app.include_router(documents_router)


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
