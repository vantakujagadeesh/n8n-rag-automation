# app/main.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/main.py
"""
AutoRAG — FastAPI Application Entry Point
==========================================
Assembles the full application:
  - CORS middleware
  - Prometheus metrics (/metrics)
  - Rate-limiting (slowapi)
  - Startup: init PostgreSQL tables + Qdrant collection
  - Shutdown: close all connection pools
  - Routers: ingest + query
  - Root endpoint + global exception handler
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import time

from app.core.config import get_settings
from app.db.postgres import close_pool, init_db
from app.models.schemas import ErrorResponse
from app.routers import ingest as ingest_router
from app.routers import query as query_router
from app.services.qdrant_service import close_client, init_collection
from app.services.redis_service import close_redis

# ── Logging setup ─────────────────────────────────────────────────────────────
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown for all connection pools and collections.

    Startup order:
      1. PostgreSQL — create tables/indexes
      2. Qdrant     — create collection + payload indexes

    Shutdown order (reverse):
      1. Qdrant client
      2. Redis client
      3. PostgreSQL pool
    """
    # ── STARTUP ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("AutoRAG API v%s — Starting up…", settings.app_version)

    try:
        logger.info("[1/2] Initialising PostgreSQL schema…")
        await init_db()
        logger.info("[1/2] PostgreSQL schema ready ✓")
    except Exception as exc:
        logger.critical("PostgreSQL init FAILED: %s", exc, exc_info=True)
        raise RuntimeError(f"PostgreSQL unavailable at startup: {exc}") from exc

    try:
        logger.info("[2/2] Initialising Qdrant collection…")
        await init_collection()
        logger.info("[2/2] Qdrant collection ready ✓")
    except Exception as exc:
        logger.critical("Qdrant init FAILED: %s", exc, exc_info=True)
        raise RuntimeError(f"Qdrant unavailable at startup: {exc}") from exc

    logger.info("AutoRAG API started successfully. 🚀")
    logger.info("=" * 60)

    yield   # Application runs here

    # ── SHUTDOWN ───────────────────────────────────────────────────────────────
    logger.info("AutoRAG API shutting down…")

    try:
        await close_client()
        logger.info("Qdrant client closed ✓")
    except Exception as exc:
        logger.warning("Error closing Qdrant client: %s", exc)

    try:
        await close_redis()
        logger.info("Redis client closed ✓")
    except Exception as exc:
        logger.warning("Error closing Redis client: %s", exc)

    try:
        await close_pool()
        logger.info("PostgreSQL pool closed ✓")
    except Exception as exc:
        logger.warning("Error closing PostgreSQL pool: %s", exc)

    logger.info("AutoRAG API shutdown complete.")


# ── Prometheus metrics setup ──────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "autorag_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "autorag_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
ACTIVE_REQUESTS = Counter(
    "autorag_active_requests_total",
    "Total active requests counter",
    ["method"],
)

# ── Rate Limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoRAG API",
    version=settings.app_version,
    description=(
        "Production RAG pipeline: ingest PDF/DOCX/URLs → hybrid vector search → "
        "cross-encoder rerank → LLM answer with citations."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Prometheus metrics middleware ─────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """Track request count and latency for Prometheus scraping."""
    method = request.method
    path = request.url.path

    # Skip /metrics itself to avoid infinite recursion
    if path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    REQUEST_COUNT.labels(
        method=method, endpoint=path, http_status=str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)

    return response


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(ingest_router.router, prefix="")   # /ingest, /documents
app.include_router(query_router.router,  prefix="")   # /query,  /health


# ── Prometheus scrape endpoint ─────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False, tags=["observability"])
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"], summary="API root / status check")
async def root() -> dict:
    """Returns basic API identity information."""
    return {
        "name":    "AutoRAG",
        "version": settings.app_version,
        "status":  "running",
        "docs":    "/docs",
    }


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches any unhandled exception that escapes route handlers.
    Logs the full traceback and returns a sanitised 500 ErrorResponse.
    """
    tb = traceback.format_exc()
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        tb,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            detail="An unexpected error occurred. Please check the server logs.",
        ).model_dump(),
    )
