# app/services/reranker.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/services/reranker.py
"""
AutoRAG — Cross-Encoder Reranker
=================================
Uses sentence-transformers CrossEncoder to rescore hybrid search candidates.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Lightweight, fast, and high-quality MRR on MS MARCO passage ranking.
  - Input: (query, passage) pair → raw logit score (no softmax cap).
  - Higher score = more relevant. Typical range: [-10, +10].

Design:
  - Model loaded lazily on first call — zero import-time overhead.
  - Singleton via module-level _model + threading.Lock (sync model load).
  - All public functions are synchronous (CrossEncoder.predict is blocking).
    Wrap in asyncio.to_thread() if calling from an async route handler.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

# sentence-transformers is installed in the venv.
# Import guard gives a clean error if somehow missing.
try:
    from sentence_transformers import CrossEncoder
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is not installed. "
        "Run: pip install sentence-transformers"
    ) from _e

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_TOP_K: int = 5
_DEFAULT_MIN_SCORE: float = -5.0

# ── Singleton ─────────────────────────────────────────────────────────────────
_model: CrossEncoder | None = None
_model_lock: threading.Lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    """
    Returns the CrossEncoder singleton, loading it on first call.

    Thread-safe via threading.Lock with double-checked locking.
    Model download (~25 MB) happens once and is cached by HuggingFace Hub.

    Returns:
        Loaded CrossEncoder instance ready for inference.

    Raises:
        RuntimeError: If the model fails to load (network, disk, or HF issue).
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-checked locking — another thread may have loaded while waiting
        if _model is not None:
            return _model

        logger.info("Loading cross-encoder model '%s' (first call)…", _MODEL_NAME)
        t0 = time.perf_counter()
        try:
            _model = CrossEncoder(
                _MODEL_NAME,
                max_length=512,   # truncate to model context window
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "Cross-encoder '%s' loaded in %.0f ms.", _MODEL_NAME, elapsed
            )
        except Exception as exc:
            logger.exception("Failed to load cross-encoder model '%s'.", _MODEL_NAME)
            raise RuntimeError(
                f"Could not load reranker model '{_MODEL_NAME}': {exc}"
            ) from exc

    return _model


# ── Core Rerank ───────────────────────────────────────────────────────────────

def _score_pairs(
    model: CrossEncoder,
    query: str,
    results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], float]]:
    """
    Internal: builds (query, text) pairs and batches them through CrossEncoder.

    Args:
        model:   Loaded CrossEncoder instance.
        query:   User query string.
        results: List of search result dicts (must have "text" key).

    Returns:
        List of (result_dict, score) tuples, one per input result.

    Raises:
        KeyError:    If any result dict is missing the "text" key.
        ValueError:  If results list is empty.
    """
    if not results:
        raise ValueError("Cannot rerank an empty results list.")

    pairs = []
    for idx, r in enumerate(results):
        if "text" not in r:
            raise KeyError(
                f"Result at index {idx} is missing required 'text' key. "
                f"Keys present: {list(r.keys())}"
            )
        pairs.append([query, r["text"]])

    t0 = time.perf_counter()
    scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()
    elapsed = (time.perf_counter() - t0) * 1000

    logger.debug(
        "CrossEncoder scored %d pairs in %.0f ms. "
        "Score range: [%.3f, %.3f].",
        len(pairs),
        elapsed,
        min(scores),
        max(scores),
    )

    return list(zip(results, scores))


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = _DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Reranks a list of search results using the cross-encoder model.

    Each result dict must have a "text" key containing the chunk text.
    A "rerank_score" key is added (or overwritten) on each returned dict.

    Args:
        query:   The original user query (or HyDE-expanded query).
        results: Candidate chunks from hybrid_search(), list of dicts.
        top_k:   Number of results to return. Clamped to len(results).

    Returns:
        Top-k result dicts sorted by rerank_score descending.
        Each dict has an additional "rerank_score": float key.

    Raises:
        ValueError: If results is empty.
        RuntimeError: If the model fails to load.
    """
    if not results:
        logger.warning("rerank() called with empty results list — returning [].")
        return []

    model = get_reranker()
    top_k = min(top_k, len(results))

    logger.debug(
        "Reranking %d candidates → top %d | query='%.80s…'",
        len(results),
        top_k,
        query,
    )

    scored = _score_pairs(model, query, results)

    # Sort descending by cross-encoder score
    scored.sort(key=lambda x: x[1], reverse=True)

    output: list[dict[str, Any]] = []
    for result, score in scored[:top_k]:
        enriched = dict(result)          # shallow copy — don't mutate input
        enriched["rerank_score"] = round(float(score), 6)
        output.append(enriched)

    logger.info(
        "rerank: %d candidates → %d returned | "
        "top score=%.4f | bottom score=%.4f | query='%.60s…'",
        len(results),
        len(output),
        output[0]["rerank_score"] if output else float("nan"),
        output[-1]["rerank_score"] if output else float("nan"),
        query,
    )
    return output


def rerank_with_threshold(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = _DEFAULT_TOP_K,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    """
    Reranks candidates and filters out those below a minimum score threshold.

    Guarantees at least 1 result is returned even if all candidates fall
    below min_score (returns the single best result in that case), so the
    LLM always receives some context.

    Args:
        query:     The original user query (or HyDE-expanded query).
        results:   Candidate chunks from hybrid_search(), list of dicts.
        top_k:     Maximum number of results to return.
        min_score: Minimum rerank_score to pass the filter. Default -5.0.
                   Typical MS-MARCO scores: irrelevant ≈ -8 to -4,
                   relevant ≈ 0 to +10.

    Returns:
        Filtered top-k result dicts sorted by rerank_score descending.
        Always returns at least 1 result.

    Raises:
        ValueError: If results is empty.
        RuntimeError: If the model fails to load.
    """
    if not results:
        logger.warning(
            "rerank_with_threshold() called with empty results list — returning []."
        )
        return []

    # First rerank all candidates (we need scores before filtering)
    all_scored = rerank(query, results, top_k=len(results))

    # Apply threshold filter
    above_threshold = [r for r in all_scored if r["rerank_score"] >= min_score]

    if not above_threshold:
        # Safety guarantee: always return at least the best result
        best = all_scored[0]
        logger.warning(
            "All %d reranked results below min_score=%.2f. "
            "Returning best result only (score=%.4f).",
            len(all_scored),
            min_score,
            best["rerank_score"],
        )
        return [best]

    # Trim to top_k after filtering
    final = above_threshold[:top_k]

    logger.info(
        "rerank_with_threshold: %d input → %d above threshold(%.2f) → %d returned.",
        len(results),
        len(above_threshold),
        min_score,
        len(final),
    )
    return final
