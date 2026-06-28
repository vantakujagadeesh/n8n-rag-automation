# app/services/redis_service.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/services/redis_service.py
"""
AutoRAG — Redis Service
========================
Async Redis client singleton providing:
  - SHA-256 file deduplication (is_duplicate / mark_ingested)
  - Query response caching (cache_query / get_cached_query)
  - Health check

All functions are async. Client is lazily initialised on first call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Key prefixes ──────────────────────────────────────────────────────────────
_PREFIX_HASH: str = "autorag:hash:"       # SHA-256 → doc_id
_PREFIX_QUERY: str = "autorag:query:"     # query hash → cached QueryResponse JSON

# ── Singleton ──────────────────────────────────────────────────────────────────
_client: Redis | None = None
_client_lock: asyncio.Lock = asyncio.Lock()


async def get_redis() -> Redis:
    """
    Returns the module-level async Redis singleton.
    Thread-safe via asyncio.Lock on first initialisation.
    """
    global _client
    if _client is not None:
        return _client

    async with _client_lock:
        if _client is not None:
            return _client

        settings = get_settings()
        logger.info("Initialising async Redis client → %s", settings.redis_url)
        _client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        # Verify connectivity immediately
        await _client.ping()
        logger.info("Redis async client ready.")
    return _client


async def close_redis() -> None:
    """
    Gracefully closes the Redis connection and resets the singleton.
    Call this during application shutdown (lifespan event).
    """
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Redis async client closed.")


# ── File hash / deduplication ─────────────────────────────────────────────────

def get_file_hash(file_bytes: bytes) -> str:
    """
    Computes the SHA-256 hex digest of raw file bytes.

    Args:
        file_bytes: Raw content of the uploaded file.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(file_bytes).hexdigest()


async def is_duplicate(file_hash: str) -> bool:
    """
    Checks whether a file with this SHA-256 hash has already been ingested.

    Args:
        file_hash: SHA-256 hex digest of the file.

    Returns:
        True if the hash exists in Redis, False otherwise.
        Returns False (allow ingest) on Redis errors to avoid blocking ingestion.
    """
    try:
        client = await get_redis()
        exists = await client.exists(_PREFIX_HASH + file_hash)
        result = bool(exists)
        logger.debug("is_duplicate(%s…): %s", file_hash[:12], result)
        return result
    except RedisError as exc:
        logger.warning(
            "Redis error in is_duplicate — treating as non-duplicate: %s", exc
        )
        return False


async def mark_ingested(file_hash: str, doc_id: str) -> None:
    """
    Records that a file has been successfully ingested.
    The hash key is stored indefinitely (no TTL) since document
    deduplication should persist for the lifetime of the knowledge base.

    Args:
        file_hash: SHA-256 hex digest of the ingested file.
        doc_id:    UUID of the document record stored in PostgreSQL.
    """
    try:
        client = await get_redis()
        await client.set(_PREFIX_HASH + file_hash, doc_id)
        logger.debug("mark_ingested: hash=%s… → doc_id=%s", file_hash[:12], doc_id)
    except RedisError as exc:
        # Non-fatal: log but don't crash the ingest pipeline.
        # The PostgreSQL record still exists as a source of truth.
        logger.warning("Redis error in mark_ingested (non-fatal): %s", exc)


# ── Query caching ─────────────────────────────────────────────────────────────

def get_query_hash(question: str) -> str:
    """
    Computes a deterministic cache key from the question string.

    The hash is derived from the normalised (lowercased, stripped) question
    so that trivial variations (e.g. trailing spaces) hit the same cache slot.

    Args:
        question: Raw user question string.

    Returns:
        32-character MD5 hex string (sufficient for cache keys; not security use).
    """
    normalised = question.lower().strip()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()  # noqa: S324


async def cache_query(question: str, response_data: dict[str, Any]) -> None:
    """
    Caches a serialised QueryResponse dict in Redis with TTL.

    Args:
        question:      The original user question (used to derive the key).
        response_data: The QueryResponse dict to cache (must be JSON-serialisable).
    """
    settings = get_settings()
    ttl = settings.query_cache_ttl

    try:
        client = await get_redis()
        key = _PREFIX_QUERY + get_query_hash(question)
        serialised = json.dumps(response_data, default=str)
        await client.setex(key, ttl, serialised)
        logger.debug(
            "cache_query: stored '%s…' (ttl=%ds, %d bytes)",
            question[:50],
            ttl,
            len(serialised),
        )
    except (RedisError, TypeError) as exc:
        # Non-fatal: cache miss on next request is acceptable.
        logger.warning("Redis error in cache_query (non-fatal): %s", exc)


async def get_cached_query(question: str) -> dict[str, Any] | None:
    """
    Returns a previously cached QueryResponse dict, or None on cache miss.

    Args:
        question: The original user question.

    Returns:
        Deserialised response dict, or None if not cached / on Redis error.
    """
    try:
        client = await get_redis()
        key = _PREFIX_QUERY + get_query_hash(question)
        raw = await client.get(key)
        if raw is None:
            logger.debug("cache_query: MISS for '%s…'", question[:50])
            return None
        data = json.loads(raw)
        logger.info("cache_query: HIT for '%s…'", question[:50])
        return data
    except (RedisError, json.JSONDecodeError) as exc:
        logger.warning("Redis error in get_cached_query (non-fatal): %s", exc)
        return None


# ── Health check ──────────────────────────────────────────────────────────────

async def health_check() -> bool:
    """
    Validates Redis connectivity via PING.

    Returns:
        True if Redis responds, False otherwise.
        Does NOT raise — safe to call from health-check endpoints.
    """
    try:
        client = await get_redis()
        pong = await client.ping()
        return pong is True
    except Exception as exc:
        logger.warning("Redis health_check failed: %s", exc)
        return False
