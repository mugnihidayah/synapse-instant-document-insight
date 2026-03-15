"""
FastAPI application for synapse RAG

This module creates and configures the FastAPI application
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from src import __version__
from src.api.rate_limiter import limiter
from src.api.routes import documents_router, insights_router, keys_router, query_router
from src.api.session import cleanup_expired_sessions
from src.core.config import settings
from src.core.logger import get_logger, setup_logging
from src.db.connection import get_db_context

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager

    Runs on startup and shutdown
    """

    # startup
    logger.info("app_started", version=__version__)
    settings.setup_environment()

    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    cleanup_task.cancel()

    # shutdown
    logger.info("app_stopped")


async def periodic_cleanup():
    """Periodically cleanup expired sessions"""
    while True:
        try:
            await asyncio.sleep(3600)
            async with get_db_context() as db:
                count = await cleanup_expired_sessions(db)
                if count > 0:
                    logger.info("expired_sessions_cleaned", count=count)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("cleanup_failed", error=str(e))


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
      Configured FastAPI app instance
    """

    app = FastAPI(
        title="Synapse RAG API",
        description="Production-ready RAG API for document Q&A",
        version=__version__,
        lifespan=lifespan,
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(keys_router, prefix="/api/v1")
    app.include_router(insights_router, prefix="/api/v1")

    return app


app = create_app()


# HEALTH CHECK
@app.get("/health")
def health_check() -> dict:
    """
    Health check endpoint

    Used by orchestrators (e.g Docker) to check if app is alive
    """
    return {
        "status": "healthy",
        "service": "synapse-rag",
        "version": __version__,
    }


@app.get("/")
def root() -> dict:
    """Root endpoint with API info"""
    return {
        "message": "Welcome to Synapse RAG API",
        "docs": "/docs",
        "health": "/health",
    }


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()

        # log request
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        # log response
        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response


app.add_middleware(LoggingMiddleware)
