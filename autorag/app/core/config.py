# app/core/config.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/core/config.py

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    AutoRAG — Central configuration loaded from .env
    All fields are validated at startup. Missing required fields raise immediately.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "documents"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: str

    # ── Slack ─────────────────────────────────────────────────────────────────
    slack_bot_token: Optional[str] = None
    slack_signing_secret: Optional[str] = None

    # ── AWS S3 ────────────────────────────────────────────────────────────────
    s3_bucket: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"

    # ── n8n ───────────────────────────────────────────────────────────────────
    n8n_webhook_base: Optional[str] = None

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072

    # ── LLM ───────────────────────────────────────────────────────────────────
    primary_llm: str = "gpt-4o"
    fallback_llm: str = "claude-sonnet-4-6"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "AutoRAG"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── RAG pipeline tuning ───────────────────────────────────────────────────
    retrieval_top_k: int = 20          # candidates fetched from Qdrant
    rerank_top_k: int = 5              # final chunks sent to LLM after reranking
    query_cache_ttl: int = 3600        # Redis cache TTL in seconds (1 hour)
    hyde_enabled: bool = True          # HyDE query expansion toggle
    hybrid_search_enabled: bool = True # dense + BM25 fusion toggle

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_s3_configured(self) -> bool:
        return all([self.s3_bucket, self.aws_access_key_id, self.aws_secret_access_key])

    @property
    def is_slack_configured(self) -> bool:
        return all([self.slack_bot_token, self.slack_signing_secret])

    @property
    def postgres_url_async(self) -> str:
        """
        asyncpg requires 'postgresql://' not 'postgres://'.
        Render sometimes returns 'postgres://' — this fixes it.
        """
        return self.database_url.replace("postgres://", "postgresql://", 1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Use this everywhere: from app.core.config import get_settings
    settings = get_settings()
    """
    return Settings()


# Module-level singleton for convenience imports
settings = get_settings()
