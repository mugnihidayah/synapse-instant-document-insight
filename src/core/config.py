"""
Configuration management using Pydantic Settings

This module provides centralized configuration class that:
- Loads settings from environment variables
- Validates settings at startup
- Provides type-safe access to configuration values

Usage:
    from src.core.config import settings

    api_key = settings.groq_api_key
    model = settings.llm_model
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_upload_dir() -> Path:
    """Prefer HF persistent storage when available."""
    data_root = Path("/data")
    if data_root.exists() and data_root.is_dir():
        return data_root / "uploads"
    return Path("./uploads")


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables

    All settings can be overriden by environment variables
    Environment variable are case-insensitive

    Example:
        Set GROQ_API_KEY in .env or environment to configure Groq API key
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = Field(
        default="",
        description="API key from Groq LLM service",
    )

    huggingface_token: str = Field(
        default="",
        description="Huggingface API token for model downloads",
    )

    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Huggingface model name for embeddings",
    )

    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default LLM model to use",
    )

    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="Flashrank model for reranking",
    )

    available_llms: list[str] = Field(
        default=[
            "llama-3.3-70b-versatile",
            "moonshotai/kimi-k2-instruct-0905",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
        ],
        description="List of available LLM models",
    )

    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=2000,
        description="Size of document chunks in characters",
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between chunks in characters",
    )

    retrieval_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of documents to retrieve before reranking",
    )

    retrieval_fetch_k: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Initial retrieval count before MMR/diversification",
    )

    dynamic_top_k_min: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Minimum dynamic top_k for retrieval",
    )

    dynamic_top_k_max: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum dynamic top_k for retrieval",
    )

    rerank_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of documents to keep after reranking",
    )

    use_mmr: bool = Field(
        default=True,
        description="Apply lightweight MMR diversification after reranking",
    )

    mmr_lambda: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Balance relevance/diversity in MMR (higher=relevance)",
    )

    groundedness_threshold: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Minimum lexical grounding score required for strict grounding mode",
    )

    query_rewrite_enabled: bool = Field(
        default=True,
        description="Enable lightweight query rewrite before retrieval",
    )

    cache_dir: Path = Field(
        default=Path("./opt"),
        description="Directory for caching models and data",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://synapse:synapse_password@localhost:5432/synapse_db",
        ),
        description="PostgreSQL connection URL",
    )

    log_level: str = Field(
        default="info",
        description="Log level (debug, info, warning, error)",
    )

    reranker_provider: str = Field(
        default="cohere",
        description="Reranker provider (cohere, local, or none)",
    )

    cohere_api_key: str = Field(default="", description="Cohere API key for reranker")

    # hybrid search
    use_hybrid_search: bool = Field(
        default=True, description="Use hybrid search (vector + keyword)"
    )
    hybrid_vector_weight: float = Field(
        default=0.5,
        description="Weight for vector search in hybrid search (0-1)",
    )
    hybrid_keyword_weight: float = Field(
        default=0.5,
        description="Weight for keyword search in hybrid search (0-1)",
    )

    ingestion_async_default: bool = Field(
        default=True,
        description="Default upload mode. True runs ingestion in background.",
    )

    enable_ocr: bool = Field(
        default=True,
        description="Enable OCR for image files and scanned PDF images.",
    )

    enable_table_extraction: bool = Field(
        default=True,
        description="Enable lightweight table extraction for PDF pages.",
    )

    max_upload_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum single file upload size in MB.",
    )

    upload_dir: Path = Field(
        default_factory=_default_upload_dir,
        description="Directory to store original uploaded files.",
    )

    usage_daily_query_quota: int = Field(
        default=1000,
        ge=1,
        le=200000,
        description="Soft quota for queries per API key per day.",
    )

    export_max_messages: int = Field(
        default=200,
        ge=10,
        le=2000,
        description="Maximum number of chat messages included in exports.",
    )

    # Agent settings
    agent_enabled: bool = Field(
        default=True,
        description="Enable agentic RAG mode",
    )

    agent_max_iterations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum reasoning steps for agentic RAG",
    )

    agent_temperature: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="LLM temperature for agent reasoning (lower = more focused)",
    )

    @field_validator("groq_api_key", "huggingface_token")
    @classmethod
    def check_not_placeholder(cls, v: str, info) -> str:
        """warn if API keys are placeholders values"""
        if v and ("your_" in v.lower() or "xxx" in v.lower()):
            import warnings

            warnings.warn(
                f"{info.field_name} appears to be a placeholder value."
                "Please set a valid API key in the environment or .env file.",
                stacklevel=2,
            )
        return v

    @field_validator("dynamic_top_k_max")
    @classmethod
    def validate_dynamic_top_k_range(cls, v: int, info) -> int:
        min_value = info.data.get("dynamic_top_k_min", 1)
        if v < min_value:
            raise ValueError("dynamic_top_k_max must be >= dynamic_top_k_min")
        return v

    @property
    def is_groq_configured(self) -> bool:
        """Check if Groq API key is configured"""
        return bool(self.groq_api_key) and "your_" not in self.groq_api_key.lower()

    @property
    def is_huggingface_configured(self) -> bool:
        """Check if Huggingface token is configured"""
        return bool(self.huggingface_token) and "your_" not in self.huggingface_token.lower()

    def setup_environment(self) -> None:
        """Setup environment variables for external libraries"""
        if self.huggingface_token:
            os.environ["HUGGINGFACE_TOKEN"] = self.huggingface_token
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Get cahced settings instance

    Use lru_cache to ensure settings are only loaded once

    Returns:
        Settings: The application settings instance
    """
    settings = Settings()
    settings.setup_environment()
    return settings


settings = get_settings()
