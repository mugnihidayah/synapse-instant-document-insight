"""
FastAPI application for synapse RAG

This module creates nad configures the FastAPI application
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import documents_router, query_router
from src.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager

    Runs on startup and shutdown
    """

    # startup
    print("starting Synapse RAG API...")
    settings.setup_environment()

    yield

    # shutdown
    print("shutting down Synapse RAG API...")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application

    Returns:
      Configured FastAPI app instance
    """

    app = FastAPI(
        title="Synapse RAG API",
        description="Production-ready RAG API for document Q&A",
        version="0.1.0",
        lifespan=lifespan,
    )

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
        "version": "0.1.0",
    }


@app.get("/")
def root() -> dict:
    """Root endpoint with API info"""
    return {
        "message": "Welcome to Synapse RAG API",
        "docs": "/docs",
        "health": "/health",
    }
