"""
API routes package
"""

from src.api.routes.documents import router as documents_router
from src.api.routes.keys import router as keys_router
from src.api.routes.query import router as query_router

__all__ = ["documents_router", "query_router", "keys_router"]
