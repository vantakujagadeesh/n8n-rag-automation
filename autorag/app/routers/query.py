# app/routers/query.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/routers/query.py
"""
AutoRAG — Query Router
POST /query   — Answer a question from the knowledge base
GET  /health  — Check all downstream service health
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.db import postgres
from app.models.schemas import (
    ChunkMetadata,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
    ServiceStatus,
)
from app.services import embedder, llm_service, qdrant_service, redis_service
from app.services.reranker import rerank

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)
settings = get_settings()


def _chunks_to_search_results(chunks: list[dict[str, Any]]) -> list[SearchResult]:
    """Converts reranked chunk dicts into SearchResult Pydantic models."""
    results: list[SearchResult] = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        results.append(
            SearchResult(
                chunk_id=chunk.get("chunk_id", ""),
                score=chunk.get("rerank_score", chunk.get("score", 0.0)),
                text=chunk.get("text", ""),
                metadata=ChunkMetadata(
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    text_preview=meta.get("text_preview", chunk.get("text", "")[:200]),
                    file_type=meta.get("file_type"),
                    source_url=meta.get("source_url"),
                ),
            )
        )
    return results


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question from the knowledge base",
    responses={
        200: {"description": "Answer generated successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal pipeline error"},
    },
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    Full RAG query pipeline:
    cache check → HyDE → embed → hybrid search → rerank → LLM → log → cache
    """
    start_time = time.time()
    question = request.question.strip()
    top_k = request.top_k

    logger.info("Query: top_k=%d question='%.80s…'", top_k, question)

    # ── 1. Redis cache check ───────────────────────────────────────────────────
    cached = await redis_service.get_cached_query(question)
    if cached is not None:
        cached["latency_ms"] = int((time.time() - start_time) * 1000)
        logger.info("Cache HIT for question='%.60s…'", question)
        try:
            return QueryResponse(**cached)
        except Exception:
            # Malformed cache entry — fall through to full pipeline
            logger.warning("Malformed cache entry — running full pipeline.")

    # ── 2. HyDE expansion + embedding ─────────────────────────────────────────
    hyde_used = False
    search_text = question

    if settings.hyde_enabled:
        logger.debug("HyDE: generating hypothetical document")
        hypothetical_doc = await llm_service.generate_hypothetical_doc(question)
        if hypothetical_doc and hypothetical_doc != question:
            hyde_used = True
            search_text = hypothetical_doc
            logger.info("HyDE expansion: %d chars → embedding", len(hypothetical_doc))

    logger.debug("Embedding query text (%d chars)", len(search_text))
    query_vector = await embedder.embed_query(search_text)

    # ── 3. Hybrid search ───────────────────────────────────────────────────────
    logger.debug("Hybrid search: top_k=20")
    if settings.hybrid_search_enabled:
        search_results = await qdrant_service.hybrid_search(
            query_vector=query_vector,
            query_text=question,  # always use original question for BM25
            top_k=20,
        )
    else:
        search_results = await qdrant_service.dense_search(
            query_vector=query_vector,
            top_k=20,
        )

    if not search_results:
        logger.warning("No search results returned for question='%.60s…'", question)
        latency_ms = int((time.time() - start_time) * 1000)
        no_result_response = QueryResponse(
            answer="I could not find relevant information in the knowledge base.",
            sources=[],
            latency_ms=latency_ms,
            token_count=0,
            model_used=settings.primary_llm,
            question=question,
            hyde_used=hyde_used,
        )
        return no_result_response

    logger.info("Search returned %d candidates", len(search_results))

    # ── 4. Cross-encoder rerank ────────────────────────────────────────────────
    logger.debug("Reranking %d candidates → top_%d", len(search_results), top_k)
    reranked = await asyncio.to_thread(rerank, question, search_results, top_k)
    logger.info("Rerank complete: %d results", len(reranked))

    # ── 5. Generate answer ─────────────────────────────────────────────────────
    logger.debug("Generating LLM answer")
    llm_result = await llm_service.generate_answer(
        question=question,
        context_chunks=reranked,
    )
    answer_text: str = llm_result["answer"]
    model_used: str = llm_result["model_used"]
    token_count: int = llm_result["token_count"]

    # ── 6. Log to PostgreSQL ───────────────────────────────────────────────────
    latency_ms = int((time.time() - start_time) * 1000)
    doc_ids = list({
        chunk.get("metadata", {}).get("doc_id", "")
        for chunk in reranked
        if chunk.get("metadata", {}).get("doc_id")
    })

    try:
        await postgres.log_query(
            question=question,
            answer=answer_text,
            doc_ids=doc_ids,
            latency_ms=latency_ms,
            token_count=token_count,
            model_used=model_used,
        )
    except Exception as exc:
        # Non-fatal — don't fail the response due to a logging error
        logger.warning("Failed to log query to PostgreSQL (non-fatal): %s", exc)

    # ── 7. Build response ──────────────────────────────────────────────────────
    sources = _chunks_to_search_results(reranked)
    response = QueryResponse(
        answer=answer_text,
        sources=sources,
        latency_ms=latency_ms,
        token_count=token_count,
        model_used=model_used,
        question=question,
        hyde_used=hyde_used,
    )

    # ── 8. Cache in Redis ──────────────────────────────────────────────────────
    try:
        cache_payload = response.model_dump(mode="json")
        await redis_service.cache_query(question, cache_payload)
    except Exception as exc:
        logger.warning("Failed to cache query response (non-fatal): %s", exc)

    logger.info(
        "Query OK: model=%s tokens=%d latency=%dms hyde=%s sources=%d",
        model_used, token_count, latency_ms, hyde_used, len(sources),
    )
    return response


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check health of all downstream services",
)
async def health_check() -> HealthResponse:
    """
    Checks Qdrant, Redis, and PostgreSQL connectivity in parallel.
    Returns overall status: 'ok' if all pass, 'degraded' if any fail.
    """
    t_start = time.time()

    async def _check_qdrant() -> tuple[str, ServiceStatus]:
        t0 = time.time()
        try:
            ok = await qdrant_service.health_check()
            latency = int((time.time() - t0) * 1000)
            return "qdrant", ServiceStatus(status="ok" if ok else "error", latency_ms=latency)
        except Exception as exc:
            latency = int((time.time() - t0) * 1000)
            return "qdrant", ServiceStatus(status="error", latency_ms=latency, detail=str(exc))

    async def _check_redis() -> tuple[str, ServiceStatus]:
        t0 = time.time()
        try:
            ok = await redis_service.health_check()
            latency = int((time.time() - t0) * 1000)
            return "redis", ServiceStatus(status="ok" if ok else "error", latency_ms=latency)
        except Exception as exc:
            latency = int((time.time() - t0) * 1000)
            return "redis", ServiceStatus(status="error", latency_ms=latency, detail=str(exc))

    async def _check_postgres() -> tuple[str, ServiceStatus]:
        t0 = time.time()
        try:
            ok = await postgres.health_check()
            latency = int((time.time() - t0) * 1000)
            return "postgres", ServiceStatus(status="ok" if ok else "error", latency_ms=latency)
        except Exception as exc:
            latency = int((time.time() - t0) * 1000)
            return "postgres", ServiceStatus(status="error", latency_ms=latency, detail=str(exc))

    results = await asyncio.gather(
        _check_qdrant(),
        _check_redis(),
        _check_postgres(),
        return_exceptions=False,
    )

    services: dict[str, ServiceStatus] = {name: svc for name, svc in results}
    all_ok = all(svc.status == "ok" for svc in services.values())
    overall = "ok" if all_ok else "degraded"

    logger.info(
        "Health check: %s | qdrant=%s redis=%s postgres=%s | %.0fms",
        overall,
        services["qdrant"].status,
        services["redis"].status,
        services["postgres"].status,
        (time.time() - t_start) * 1000,
    )

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        services=services,
        timestamp=datetime.utcnow(),
    )
