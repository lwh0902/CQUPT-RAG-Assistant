"""FastAPI router package."""

from routers.auth import router as auth_router
from routers.chat import router as chat_router
from routers.documents import router as documents_router
from routers.memories import router as memories_router
from routers.sessions import router as sessions_router
from routers.settings import router as settings_router

__all__ = ["auth_router", "chat_router", "documents_router", "memories_router", "sessions_router", "settings_router"]
