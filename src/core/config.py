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

import warnings
from email.policy import default
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  """
  Application settings loaded from environment variables

  All settings can be overriden by environment variables
  Environment variable are case-insensitive

  Example:
    Set GROQ_API_KEY in .env or environment to configure Groq API key 
  """

  model_config = SettingsConfigDict(
    env_file = ".env",
    env_file_encoding = "utf-8",
    case_sensitive = False,
    extra = "ignore",
  )

  groq_api_key: str = Field(
    default = "",
    description = "API key from Groq LLM service",
  )

  huggingface_token: str = Field(
    default = "",
    description = "Huggingface API token for model downloads",
  )

  embedding_model: str = Field(
    default = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    description = "Huggingface model name for embeddings",
  )

  llm_model: str = Field(
    default = "llama-3.3-70b-versatile",
    description = "Default LLM model to use",
  )

  reranker_model: str = Field(
    default = "ms-marco-MiniLM-L-12-v2",
    description = "Flashrank model for reranking",
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
    default = 500,
    ge = 100,
    le = 2000,
    description = "Size of document chunks in characters",
  )

  chunk_overlap: int = Field(
    default = 100,
    ge = 0,
    le = 500,
    description = "Overlap between chunks in characters",
  )

  retrieval_top_k: int = Field(
    default = 10,
    ge = 1,
    le = 50,
    description = "Number of documents to retrieve before reranking",
  )

  rerank_top_k: int = Field(
    default = 3,
    ge = 1,
    le = 10,
    description = "Number of documents to keep after reranking",
  )

  cache_dir: Path = Field(
    default = Path("./opt"),
    description = "Directory for caching models and data",
  )

  debug: bool = Field(
    default = False,
    description = "Enable debug mode",
  )

  @field_validator("groq_api_key", "huggingface_token")
  @classmethod
  def check_not_placeholder(cls, v: str, info) -> str:
    """warn if API keys are placeholders values"""
    if v and ("your_" in v.lower() or "xxx" in v.lower()):
      import warnings
      warnings.warn(
        f"{info.field_name} appears to be a placeholder value."
        "Please set a valid API key in the environment or .env file."
      )
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