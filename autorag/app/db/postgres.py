# app/db/postgres.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/db/postgres.py

import logging
from typing import Optional
from datetime import datetime

import asyncpg
from asyncpg import Pool, Connection

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton pool ─────────────────────────────────────────────────────────────
_pool: Optional[Pool] = None


# ── DDL ───────────────────────────────────────────────────────────────────────
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT        NOT NULL,
    file_hash   TEXT UNIQUE NOT NULL,
    chunk_count INTEGER     NOT NULL,
    file_type   TEXT,
    source_url  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_QUERY_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS query_logs (
    id          SERIAL PRIMARY KEY,
    question    TEXT        NOT NULL,
    answer      TEXT        NOT NULL,
    doc_ids     TEXT[],
    latency_ms  INTEGER,
    token_count INTEGER,
    model_used  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);",
    "CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at DESC);",
]


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

async def get_pool() -> Pool:
    """
    Returns the singleton asyncpg connection pool.
    Creates it on first call. Subsequent calls return the cached pool.
    """
    global _pool
    if _pool is None:
        logger.info("Creating asyncpg connection pool...")
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.postgres_url_async,
                min_size=2,
                max_size=10,
                command_timeout=30,
                server_settings={"application_name": "autorag"},
            )
            logger.info("PostgreSQL connection pool created (min=2, max=10)")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL pool: {e}")
            raise
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool. Call on app shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


# ── Schema init ───────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Creates all tables and indexes if they don't exist.
    Safe to call multiple times (idempotent).
    Call this in FastAPI startup event.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        conn: Connection
        async with conn.transaction():
            await conn.execute(CREATE_DOCUMENTS_TABLE)
            logger.info("Table 'documents' ready")

            await conn.execute(CREATE_QUERY_LOGS_TABLE)
            logger.info("Table 'query_logs' ready")

            for index_sql in CREATE_INDEXES:
                await conn.execute(index_sql)

        logger.info("Database schema initialised successfully")


# ── Document operations ───────────────────────────────────────────────────────

async def get_document_by_hash(file_hash: str) -> Optional[dict]:
    """
    Returns the document row if a file with this SHA-256 hash was already ingested.
    Used for Redis-backed deduplication fallback.

    Returns:
        dict with keys (id, filename, chunk_count) or None if not found.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, filename, chunk_count, created_at
            FROM documents
            WHERE file_hash = $1
            """,
            file_hash,
        )
        if row:
            return dict(row)
        return None


async def save_document(
    doc_id: str,
    filename: str,
    file_hash: str,
    chunk_count: int,
    file_type: Optional[str] = None,
    source_url: Optional[str] = None,
) -> None:
    """
    Inserts a newly ingested document record into the documents table.

    Args:
        doc_id:      UUID string generated at ingest time
        filename:    original filename (e.g. "policy.pdf")
        file_hash:   SHA-256 hex digest of raw file bytes
        chunk_count: number of chunks stored in Qdrant
        file_type:   "pdf" | "docx" | "url" | None
        source_url:  original URL if ingested from web
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, filename, file_hash, chunk_count, file_type, source_url)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (file_hash) DO NOTHING
            """,
            doc_id,
            filename,
            file_hash,
            chunk_count,
            file_type,
            source_url,
        )
        logger.debug(f"Document saved: {doc_id} ({filename}, {chunk_count} chunks)")


async def get_document_by_id(doc_id: str) -> Optional[dict]:
    """Returns a document record by its primary key ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM documents WHERE id = $1",
            doc_id,
        )
        return dict(row) if row else None


async def list_documents(limit: int = 50, offset: int = 0) -> list[dict]:
    """Returns paginated list of all ingested documents."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, chunk_count, file_type, created_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return [dict(row) for row in rows]


# ── Query log operations ──────────────────────────────────────────────────────

async def log_query(
    question: str,
    answer: str,
    doc_ids: list[str],
    latency_ms: int,
    token_count: int,
    model_used: str,
) -> int:
    """
    Logs every /query call to PostgreSQL for observability and eval.

    Returns:
        The auto-generated log row ID.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO query_logs
                (question, answer, doc_ids, latency_ms, token_count, model_used)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            question,
            answer,
            doc_ids,
            latency_ms,
            token_count,
            model_used,
        )
        logger.debug(f"Query logged: id={row_id}, latency={latency_ms}ms, tokens={token_count}")
        return row_id


async def get_query_stats(days: int = 7) -> dict:
    """
    Returns aggregate query stats for the last N days.
    Used by the n8n eval cron weekly Slack report.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)                        AS total_queries,
                ROUND(AVG(latency_ms))          AS avg_latency_ms,
                ROUND(AVG(token_count))         AS avg_tokens,
                MIN(created_at)                 AS from_date,
                MAX(created_at)                 AS to_date
            FROM query_logs
            WHERE created_at >= NOW() - ($1 || ' days')::INTERVAL
            """,
            str(days),
        )
        return dict(row) if row else {}


async def health_check() -> bool:
    """
    Runs a lightweight SELECT 1 to confirm PostgreSQL is reachable.
    Used by GET /health endpoint.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        return False
