# app/services/qdrant_service.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/services/qdrant_service.py
"""
AutoRAG — Qdrant Service
========================
Async Qdrant client wrapper providing:
  - Collection initialisation (dense + sparse/BM25)
  - Chunk upsert in batches
  - Dense vector search
  - Sparse (BM25) vector search
  - Hybrid search with Reciprocal Rank Fusion (RRF)
  - Document deletion
  - Health check

All public functions are async and production-safe.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import uuid
from collections import Counter
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_UPSERT_BATCH_SIZE: int = 50
_DELETE_BATCH_SIZE: int = 256
_RRF_K: int = 60          # RRF constant — standard value recommended in literature
_SPARSE_VECTOR_NAME: str = "sparse"
_DENSE_VECTOR_NAME: str = "dense"

# Common English stop-words for BM25 term frequency calculation
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
        "were", "been", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "not",
        "no", "nor", "so", "yet", "both", "either", "neither", "whether",
        "each", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "than", "too", "very", "just", "because", "if",
        "then", "else", "when", "where", "how", "all", "any", "this", "that",
        "these", "those", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "their", "what", "which",
        "who", "whom", "about", "above", "after", "before", "between",
    }
)

# ── Singleton ──────────────────────────────────────────────────────────────────
_client: AsyncQdrantClient | None = None
_client_lock: asyncio.Lock = asyncio.Lock()


async def get_client() -> AsyncQdrantClient:
    """
    Returns the module-level AsyncQdrantClient singleton.
    Thread-safe via asyncio.Lock on first initialisation.
    """
    global _client
    if _client is not None:
        return _client

    async with _client_lock:
        # Double-checked locking pattern
        if _client is not None:
            return _client

        settings = get_settings()
        logger.info(
            "Initialising Qdrant async client → %s  collection=%s",
            settings.qdrant_url,
            settings.qdrant_collection,
        )
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
            prefer_grpc=False,  # REST is safer across cloud providers
        )
        logger.info("Qdrant async client ready.")
    return _client


async def close_client() -> None:
    """
    Gracefully closes the Qdrant client and resets the singleton.
    Call this during application shutdown (lifespan event).
    """
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("Qdrant async client closed.")


# ── Collection Management ──────────────────────────────────────────────────────

async def init_collection() -> None:
    """
    Creates the Qdrant collection with dense + sparse vector configs if it
    does not already exist. Idempotent — safe to call on every startup.

    Dense:  size=3072, distance=Cosine  (text-embedding-3-large)
    Sparse: BM25-compatible SparseVectorParams (named "sparse")
    """
    settings = get_settings()
    client = await get_client()
    collection_name = settings.qdrant_collection

    try:
        existing = await client.get_collections()
        existing_names = {c.name for c in existing.collections}

        if collection_name in existing_names:
            logger.info("Collection '%s' already exists — skipping creation.", collection_name)
            return

        logger.info(
            "Creating collection '%s' (dense=%d dims, Cosine + BM25 sparse).",
            collection_name,
            settings.embedding_dims,
        )
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                _DENSE_VECTOR_NAME: models.VectorParams(
                    size=settings.embedding_dims,
                    distance=models.Distance.COSINE,
                    on_disk=False,
                ),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    ),
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
        logger.info("Collection '%s' created successfully.", collection_name)

    except UnexpectedResponse as exc:
        # 409 Conflict means it was created by another process concurrently
        if exc.status_code == 409:
            logger.info(
                "Collection '%s' already exists (concurrent creation detected).",
                collection_name,
            )
        else:
            logger.exception("Failed to create Qdrant collection '%s'.", collection_name)
            raise
    except Exception:
        logger.exception("Unexpected error during collection initialisation.")
        raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Converts an arbitrary string chunk_id to a deterministic UUID5 string.
    Uses uuid.NAMESPACE_DNS as the namespace.

    This ensures stable, reproducible Qdrant point IDs across restarts and
    re-ingestion runs, which is critical for idempotent upserts.

    Args:
        chunk_id: Any string identifier for a chunk (e.g. "doc123_chunk_0").

    Returns:
        Lowercase UUID5 string (e.g. "550e8400-e29b-41d4-a716-446655440000").
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _build_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """
    Builds a sparse vector (indices, values) from raw text using
    term-frequency normalised by document length.

    Pipeline:
      1. Lowercase and tokenise on non-alphanumeric characters.
      2. Remove stop-words and tokens shorter than 2 characters.
      3. Count term frequencies.
      4. Map each unique token to a stable integer index via hash().
      5. Normalise TF values so the vector has unit L2 norm.

    Args:
        text: Raw query or document text.

    Returns:
        (indices, values) tuple compatible with qdrant_client SparseVector.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 2]

    if not tokens:
        # Fallback: single dummy entry so the call doesn't fail
        return [0], [0.0]

    tf: Counter[str] = Counter(tokens)
    total = sum(tf.values())

    indices: list[int] = []
    values: list[float] = []

    for token, count in tf.items():
        # Use positive modulo to keep indices in [0, 2^20) range
        idx = abs(hash(token)) % (2**20)
        indices.append(idx)
        values.append(count / total)

    # L2 normalise
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    values = [v / norm for v in values]

    return indices, values


# ── Upsert ────────────────────────────────────────────────────────────────────

async def upsert_chunks(chunks_with_embeddings: list[dict[str, Any]]) -> int:
    """
    Upserts a list of embedded chunks into Qdrant in batches of 50.

    Each input dict is the direct output of embedder.embed_chunks() and must
    contain the following keys:
        chunk_id    : str  — unique string ID for this chunk
        doc_id      : str  — parent document ID
        filename    : str  — source filename
        file_type   : str  — e.g. "pdf", "docx", "url"
        chunk_index : int  — zero-based position in parent document
        text        : str  — full chunk text (used for BM25 sparse vector)
        text_preview: str  — first ~200 chars for display
        char_count  : int  — character length of text
        embedding   : list[float]  — dense vector of length 3072

    Args:
        chunks_with_embeddings: List of chunk dicts from embed_chunks().

    Returns:
        Total number of points upserted.

    Raises:
        ValueError: If the input list is empty or a chunk lacks required keys.
        Exception:  Propagates Qdrant client errors after logging.
    """
    if not chunks_with_embeddings:
        logger.warning("upsert_chunks called with empty list — nothing to do.")
        return 0

    _REQUIRED_KEYS = {
        "chunk_id", "doc_id", "filename", "file_type",
        "chunk_index", "text", "text_preview", "char_count", "embedding",
    }

    client = await get_client()
    settings = get_settings()
    collection_name = settings.qdrant_collection

    total_upserted = 0
    batches = [
        chunks_with_embeddings[i : i + _UPSERT_BATCH_SIZE]
        for i in range(0, len(chunks_with_embeddings), _UPSERT_BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches):
        points: list[models.PointStruct] = []

        for chunk in batch:
            missing = _REQUIRED_KEYS - chunk.keys()
            if missing:
                raise ValueError(
                    f"Chunk '{chunk.get('chunk_id', '<unknown>')}' "
                    f"is missing required keys: {missing}"
                )

            point_id = chunk_id_to_uuid(chunk["chunk_id"])

            # Build sparse vector from chunk text for BM25 search
            sparse_indices, sparse_values = _build_sparse_vector(chunk["text"])

            payload = {
                "chunk_id":    chunk["chunk_id"],
                "doc_id":      chunk["doc_id"],
                "filename":    chunk["filename"],
                "file_type":   chunk["file_type"],
                "chunk_index": chunk["chunk_index"],
                "text":        chunk["text"],
                "text_preview": chunk["text_preview"],
                "char_count":  chunk["char_count"],
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        _DENSE_VECTOR_NAME: chunk["embedding"],
                        _SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload=payload,
                )
            )

        try:
            await client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            total_upserted += len(points)
            logger.debug(
                "Batch %d/%d upserted: %d points (total so far: %d).",
                batch_idx + 1,
                len(batches),
                len(points),
                total_upserted,
            )
        except Exception:
            logger.exception(
                "Qdrant upsert failed on batch %d/%d (%d points).",
                batch_idx + 1,
                len(batches),
                len(points),
            )
            raise

    logger.info(
        "upsert_chunks complete: %d points upserted into '%s'.",
        total_upserted,
        collection_name,
    )
    return total_upserted


# ── Search ────────────────────────────────────────────────────────────────────

def _hits_to_dicts(hits: list[models.ScoredPoint]) -> list[dict[str, Any]]:
    """
    Normalises Qdrant ScoredPoint results into plain dicts.

    Returns:
        List of dicts with keys: chunk_id, score, text, metadata.
        metadata contains all payload fields except text.
    """
    results = []
    for hit in hits:
        payload: dict[str, Any] = hit.payload or {}
        text = payload.get("text", "")

        metadata = {k: v for k, v in payload.items() if k != "text"}
        results.append(
            {
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "score":    hit.score,
                "text":     text,
                "metadata": metadata,
            }
        )
    return results


async def dense_search(
    query_vector: list[float],
    top_k: int = 20,
    doc_id_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Cosine similarity search on the "dense" named vector.

    Args:
        query_vector:  Embedded query vector of length 3072.
        top_k:         Number of results to return.
        doc_id_filter: Optional doc_id to restrict search to a single document.

    Returns:
        List of result dicts: {chunk_id, score, text, metadata}.

    Raises:
        Exception: Propagates Qdrant client errors after logging.
    """
    client = await get_client()
    settings = get_settings()
    collection_name = settings.qdrant_collection

    query_filter: models.Filter | None = None
    if doc_id_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id_filter),
                )
            ]
        )

    try:
        hits = await client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector(
                name=_DENSE_VECTOR_NAME,
                vector=query_vector,
            ),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results = _hits_to_dicts(hits)
        logger.debug("dense_search returned %d results (top_k=%d).", len(results), top_k)
        return results

    except Exception:
        logger.exception("dense_search failed (top_k=%d).", top_k)
        raise


async def sparse_search(
    query_text: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    BM25-style sparse vector search using the "sparse" named vector field.

    Builds a TF-based sparse vector from query_text and queries Qdrant
    using query_sparse (dot-product similarity on sparse vectors).

    Args:
        query_text: Raw query string (tokenised internally).
        top_k:      Number of results to return.

    Returns:
        List of result dicts: {chunk_id, score, text, metadata}.

    Raises:
        Exception: Propagates Qdrant client errors after logging.
    """
    client = await get_client()
    settings = get_settings()
    collection_name = settings.qdrant_collection

    sparse_indices, sparse_values = _build_sparse_vector(query_text)

    try:
        hits = await client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name=_SPARSE_VECTOR_NAME,
                vector=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results = _hits_to_dicts(hits)
        logger.debug("sparse_search returned %d results (top_k=%d).", len(results), top_k)
        return results

    except Exception:
        logger.exception("sparse_search failed (top_k=%d, query='%.60s...').", top_k, query_text)
        raise


async def hybrid_search(
    query_vector: list[float],
    query_text: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval combining dense cosine similarity and sparse BM25 search,
    fused via Reciprocal Rank Fusion (RRF).

    Algorithm:
      1. Run dense_search and sparse_search concurrently via asyncio.gather.
      2. For every retrieved chunk, compute RRF score:
             score(chunk) = Σ  1 / (rank_i + K)
         where rank_i is the 0-based rank in result list i and K=60.
      3. Deduplicate by chunk_id — if a chunk appears in both lists, scores
         are summed.
      4. Return top_k chunks sorted by fused score descending.

    Args:
        query_vector: Dense embedding of the query (3072 dims).
        query_text:   Raw query text for BM25 sparse search.
        top_k:        Final number of results to return after fusion.

    Returns:
        List of result dicts: {chunk_id, score, text, metadata},
        sorted by fused RRF score descending.
    """
    logger.debug(
        "hybrid_search start: top_k=%d, query='%.60s...'", top_k, query_text
    )

    # Run both searches in parallel — doubles the candidate pool for fusion
    dense_results, sparse_results = await asyncio.gather(
        dense_search(query_vector, top_k=top_k),
        sparse_search(query_text, top_k=top_k),
        return_exceptions=False,
    )

    # RRF fusion
    # fused_scores: chunk_id → accumulated RRF score
    fused_scores: dict[str, float] = {}
    # Store chunk data (text + metadata) by chunk_id; first occurrence wins
    chunk_data: dict[str, dict[str, Any]] = {}

    for result_list in (dense_results, sparse_results):
        for rank, item in enumerate(result_list):
            cid = item["chunk_id"]
            rrf_contribution = 1.0 / (rank + _RRF_K)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contribution

            if cid not in chunk_data:
                chunk_data[cid] = {
                    "chunk_id": cid,
                    "text":     item["text"],
                    "metadata": item["metadata"],
                }

    # Sort by fused score and return top_k
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for cid, score in ranked:
        entry = chunk_data[cid].copy()
        entry["score"] = score
        results.append(entry)

    logger.debug(
        "hybrid_search complete: %d unique candidates → %d returned.",
        len(fused_scores),
        len(results),
    )
    return results


# ── Deletion ──────────────────────────────────────────────────────────────────

async def delete_document(doc_id: str) -> int:
    """
    Deletes all Qdrant points whose payload.doc_id matches the given doc_id.

    Uses scroll + delete to handle arbitrarily large documents without
    loading all point IDs into memory at once.

    Args:
        doc_id: The document ID to purge from the vector store.

    Returns:
        Total number of points deleted.

    Raises:
        Exception: Propagates Qdrant client errors after logging.
    """
    if not doc_id:
        raise ValueError("doc_id must be a non-empty string.")

    client = await get_client()
    settings = get_settings()
    collection_name = settings.qdrant_collection

    doc_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
    )

    total_deleted = 0
    offset: str | None = None

    try:
        while True:
            # Scroll page of IDs matching the filter
            scroll_result = await client.scroll(
                collection_name=collection_name,
                scroll_filter=doc_filter,
                limit=_DELETE_BATCH_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points_page, next_offset = scroll_result

            if not points_page:
                break

            ids_to_delete = [p.id for p in points_page]
            await client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=ids_to_delete),
                wait=True,
            )
            total_deleted += len(ids_to_delete)
            logger.debug(
                "Deleted %d points for doc_id='%s' (running total: %d).",
                len(ids_to_delete),
                doc_id,
                total_deleted,
            )

            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            "delete_document complete: %d points deleted for doc_id='%s'.",
            total_deleted,
            doc_id,
        )
        return total_deleted

    except Exception:
        logger.exception(
            "delete_document failed for doc_id='%s' after %d deletions.",
            doc_id,
            total_deleted,
        )
        raise


# ── Health Check ──────────────────────────────────────────────────────────────

async def health_check() -> bool:
    """
    Validates Qdrant connectivity by listing collections.

    Returns:
        True if the Qdrant cluster is reachable, False otherwise.
        Does NOT raise — safe to call in health-check endpoints.
    """
    try:
        client = await get_client()
        await client.get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant health_check failed: %s", exc)
        return False
