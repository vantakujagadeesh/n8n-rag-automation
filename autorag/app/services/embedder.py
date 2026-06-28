# app/services/embedder.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/services/embedder.py
#
# Embedding service — wraps OpenAI text-embedding-3-large.
# Handles batching, retries, and chunk dict enrichment.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openai import AsyncOpenAI, RateLimitError

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Constants ──────────────────────────────────────────────────────────────────
_BATCH_SIZE: int = 100          # OpenAI recommends ≤ 2048 inputs; 100 is safe
_MAX_RETRIES: int = 3           # Retry on RateLimitError
_BASE_BACKOFF_S: float = 1.0    # Initial backoff in seconds (doubles each retry)

# Lazily initialised singleton client
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Return (or create) the shared AsyncOpenAI client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


# ── Core embedding call ────────────────────────────────────────────────────────

async def _embed_batch(texts: list[str], attempt: int = 0) -> list[list[float]]:
    """
    Call the OpenAI Embeddings API for a single batch of texts.

    Retries up to _MAX_RETRIES times on RateLimitError using
    exponential backoff (1 s, 2 s, 4 s).

    Args:
        texts:   List of strings to embed (max _BATCH_SIZE).
        attempt: Current retry attempt (0-indexed).

    Returns:
        List of embedding vectors in the same order as the input texts.

    Raises:
        RateLimitError: If all retries are exhausted.
        Exception:      On any other API error.
    """
    client = _get_client()
    try:
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
            dimensions=settings.embedding_dims,
        )
        # API returns results sorted by index — sort defensively
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]

    except RateLimitError as exc:
        if attempt >= _MAX_RETRIES:
            logger.error(
                "RateLimitError after %d retries — giving up. Error: %s",
                _MAX_RETRIES, exc,
            )
            raise
        backoff = _BASE_BACKOFF_S * (2 ** attempt)
        logger.warning(
            "RateLimitError on batch (attempt %d/%d) — backing off %.1fs",
            attempt + 1, _MAX_RETRIES, backoff,
        )
        await asyncio.sleep(backoff)
        return await _embed_batch(texts, attempt=attempt + 1)

    except Exception as exc:
        logger.error("Embedding API error on batch of %d texts: %s", len(texts), exc)
        raise


# ── Public API ─────────────────────────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI text-embedding-3-large.

    Automatically splits into batches of _BATCH_SIZE to avoid API limits.
    Each batch is called sequentially with retry logic.

    Args:
        texts: List of strings to embed. May be any length.

    Returns:
        List of 3072-dimensional embedding vectors in the same order as input.
        Returns an empty list if input is empty.
    """
    if not texts:
        logger.debug("embed_texts called with empty list — returning []")
        return []

    total = len(texts)
    batches = [texts[i : i + _BATCH_SIZE] for i in range(0, total, _BATCH_SIZE)]
    logger.info(
        "Embedding %d texts in %d batch(es) [model=%s, dims=%d]",
        total, len(batches), settings.embedding_model, settings.embedding_dims,
    )

    all_embeddings: list[list[float]] = []
    t_start = time.time()

    for batch_idx, batch in enumerate(batches):
        logger.debug("Embedding batch %d/%d (%d texts)", batch_idx + 1, len(batches), len(batch))
        batch_embeddings = await _embed_batch(batch)
        all_embeddings.extend(batch_embeddings)

    elapsed = time.time() - t_start
    logger.info(
        "Embedding complete: %d texts → %d vectors in %.2fs",
        total, len(all_embeddings), elapsed,
    )

    # Sanity check
    assert len(all_embeddings) == total, (
        f"Embedding count mismatch: got {len(all_embeddings)}, expected {total}"
    )

    return all_embeddings


async def embed_query(text: str) -> list[float]:
    """
    Embed a single query string.

    Used at query time to convert user questions (and HyDE-expanded
    hypothetical documents) into vectors for Qdrant search.

    Args:
        text: Query string to embed.

    Returns:
        Single embedding vector of length settings.embedding_dims (3072).

    Raises:
        ValueError: If text is empty.
        Exception:  On API failure after retries.
    """
    if not text or not text.strip():
        raise ValueError("embed_query: text must be a non-empty string")

    logger.debug("Embedding query (%d chars)", len(text))
    embeddings = await embed_texts([text])
    vector = embeddings[0]

    logger.debug("Query embedded: %d dims", len(vector))
    return vector


async def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Embed all chunks from chunker.chunk_document() and attach the
    embedding vector to each chunk dict under the "embedding" key.

    Processes all chunk texts in a single batched embed_texts call
    to minimise API round-trips.

    Args:
        chunks: List of chunk dicts as returned by chunker.chunk_document().
                Each dict must have a "text" key.

    Returns:
        Same list of dicts, each with an added "embedding" key containing
        a list of 3072 floats. Returns an empty list if input is empty.

    Raises:
        KeyError:  If a chunk dict is missing the "text" key.
        Exception: On embedding API failure.
    """
    if not chunks:
        logger.debug("embed_chunks called with empty list — returning []")
        return []

    logger.info("Embedding %d chunks…", len(chunks))

    # Extract texts preserving order
    texts = [chunk["text"] for chunk in chunks]

    # Batch embed all texts in one shot
    embeddings = await embed_texts(texts)

    # Attach embeddings back to chunk dicts
    enriched: list[dict[str, Any]] = []
    for chunk, embedding in zip(chunks, embeddings):
        enriched_chunk = dict(chunk)           # shallow copy — don't mutate caller's data
        enriched_chunk["embedding"] = embedding
        enriched.append(enriched_chunk)

    logger.info(
        "Chunks embedded: %d chunks, each with %d-dim vector",
        len(enriched), len(enriched[0]["embedding"]) if enriched else 0,
    )
    return enriched
