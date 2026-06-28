"""
tests/conftest.py
=================
Shared pytest fixtures for AutoRAG test suite.

Provides:
  - async_client: httpx AsyncClient wrapping the FastAPI app
  - mock_settings: patched environment with dummy keys
  - mock_qdrant / mock_redis / mock_postgres: patched service clients
  - sample_chunk_dicts: pre-built chunk payloads for embedder/Qdrant tests
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# ── Tell pytest-asyncio to use a single event loop per session ─────────────────
@pytest.fixture(scope="session")
def event_loop():
    """Override default event_loop to use session scope."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Fake environment settings ──────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def mock_settings(tmp_path_factory):
    """
    Patch all config settings with safe dummy values so no real
    API keys / database URLs are required during unit tests.
    """
    env_vars = {
        "OPENAI_API_KEY": "sk-test-openai",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "qdrant-test-key",
        "REDIS_URL": "redis://localhost:6379/0",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/testdb",
        "QDRANT_COLLECTION": "test_documents",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "EMBEDDING_DIMS": "3072",
        "PRIMARY_LLM": "gpt-4o",
        "CHUNK_SIZE": "512",
        "CHUNK_OVERLAP": "50",
        "HYDE_ENABLED": "false",
        "HYBRID_SEARCH_ENABLED": "false",
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict("os.environ", env_vars, clear=False):
        yield


# ── Sample fixture data ────────────────────────────────────────────────────────
@pytest.fixture
def sample_text() -> str:
    """A realistic ~1000 char document excerpt for chunking tests."""
    return (
        "Artificial intelligence is transforming every industry. "
        "Machine learning models can now generate text, images, and code. "
        "Retrieval-augmented generation (RAG) combines the power of large language "
        "models with external knowledge bases to improve answer accuracy and reduce "
        "hallucinations. By indexing documents as vector embeddings and retrieving "
        "relevant context at query time, RAG systems can answer complex questions "
        "grounded in factual source material. "
        "The pipeline typically involves: document parsing, text chunking, embedding "
        "via a transformer model, storing vectors in a vector database (like Qdrant), "
        "and at query time: embedding the user question, performing hybrid search, "
        "reranking with a cross-encoder, and finally generating a cited answer with "
        "a large language model such as GPT-4o or Claude Sonnet. "
        "This approach dramatically improves factual accuracy, allows real-time "
        "knowledge updates without model fine-tuning, and provides transparent "
        "citations for every generated answer."
    )


@pytest.fixture
def sample_chunk_dicts() -> list[dict]:
    """Pre-built chunk dicts as returned by chunker.chunk_document()."""
    return [
        {
            "chunk_id": "doc-1_chunk_0",
            "doc_id": "doc-1",
            "filename": "test.pdf",
            "file_type": "pdf",
            "chunk_index": 0,
            "text": "Artificial intelligence is transforming every industry with machine learning.",
            "text_preview": "Artificial intelligence is transforming every industry with machine learning.",
            "char_count": 75,
        },
        {
            "chunk_id": "doc-1_chunk_1",
            "doc_id": "doc-1",
            "filename": "test.pdf",
            "file_type": "pdf",
            "chunk_index": 1,
            "text": "RAG combines large language models with external knowledge bases.",
            "text_preview": "RAG combines large language models with external knowledge bases.",
            "char_count": 65,
        },
    ]


@pytest.fixture
def sample_chunk_dicts_with_embeddings(sample_chunk_dicts) -> list[dict]:
    """Chunk dicts enriched with fake 3072-dim embedding vectors."""
    enriched = []
    for i, chunk in enumerate(sample_chunk_dicts):
        c = dict(chunk)
        c["embedding"] = [float(i) * 0.001] * 3072
        enriched.append(c)
    return enriched


# ── Mock external service clients ──────────────────────────────────────────────
@pytest.fixture
def mock_openai_client():
    """Patch the OpenAI client used by embedder.py."""
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [
        MagicMock(index=0, embedding=[0.1] * 3072),
    ]

    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_embedding_response

    with patch("app.services.embedder._client", mock_client):
        yield mock_client


@pytest.fixture
def mock_qdrant():
    """Patch the qdrant_service module so no real Qdrant is needed."""
    with patch("app.services.qdrant_service.upsert_chunks", new_callable=AsyncMock) as m_upsert, \
         patch("app.services.qdrant_service.dense_search", new_callable=AsyncMock) as m_dense, \
         patch("app.services.qdrant_service.hybrid_search", new_callable=AsyncMock) as m_hybrid, \
         patch("app.services.qdrant_service.health_check", new_callable=AsyncMock) as m_health, \
         patch("app.services.qdrant_service.delete_document", new_callable=AsyncMock) as m_delete, \
         patch("app.services.qdrant_service.init_collection", new_callable=AsyncMock) as m_init:

        m_upsert.return_value = 2
        m_dense.return_value = []
        m_hybrid.return_value = []
        m_health.return_value = True
        m_delete.return_value = 0
        m_init.return_value = None

        yield {
            "upsert": m_upsert,
            "dense": m_dense,
            "hybrid": m_hybrid,
            "health": m_health,
            "delete": m_delete,
            "init": m_init,
        }


@pytest.fixture
def mock_redis():
    """Patch the redis_service module so no real Redis is needed."""
    with patch("app.services.redis_service.is_duplicate", new_callable=AsyncMock) as m_dup, \
         patch("app.services.redis_service.mark_ingested", new_callable=AsyncMock) as m_mark, \
         patch("app.services.redis_service.get_cached_query", new_callable=AsyncMock) as m_get, \
         patch("app.services.redis_service.cache_query", new_callable=AsyncMock) as m_cache, \
         patch("app.services.redis_service.health_check", new_callable=AsyncMock) as m_health, \
         patch("app.services.redis_service.close_redis", new_callable=AsyncMock) as m_close:

        m_dup.return_value = False
        m_mark.return_value = None
        m_get.return_value = None  # No cache hit by default
        m_cache.return_value = None
        m_health.return_value = True
        m_close.return_value = None

        # Patch get_file_hash (sync)
        with patch("app.services.redis_service.get_file_hash", return_value="abc123hash") as m_hash:
            yield {
                "is_duplicate": m_dup,
                "mark_ingested": m_mark,
                "get_cached_query": m_get,
                "cache_query": m_cache,
                "health": m_health,
                "close": m_close,
                "get_file_hash": m_hash,
            }


@pytest.fixture
def mock_postgres():
    """Patch postgres module so no real PostgreSQL is needed."""
    with patch("app.db.postgres.init_db", new_callable=AsyncMock) as m_init, \
         patch("app.db.postgres.close_pool", new_callable=AsyncMock) as m_close, \
         patch("app.db.postgres.save_document", new_callable=AsyncMock) as m_save, \
         patch("app.db.postgres.list_documents", new_callable=AsyncMock) as m_list, \
         patch("app.db.postgres.log_query", new_callable=AsyncMock) as m_log, \
         patch("app.db.postgres.health_check", new_callable=AsyncMock) as m_health:

        m_init.return_value = None
        m_close.return_value = None
        m_save.return_value = None
        m_list.return_value = []
        m_log.return_value = None
        m_health.return_value = True

        yield {
            "init": m_init,
            "close": m_close,
            "save": m_save,
            "list": m_list,
            "log": m_log,
            "health": m_health,
        }


@pytest.fixture
def mock_embedder():
    """Patch embedder so no real OpenAI calls are made."""
    fake_vector = [0.42] * 3072
    with patch("app.services.embedder.embed_query", new_callable=AsyncMock) as m_query, \
         patch("app.services.embedder.embed_chunks", new_callable=AsyncMock) as m_chunks, \
         patch("app.services.embedder.embed_texts", new_callable=AsyncMock) as m_texts:

        m_query.return_value = fake_vector
        m_chunks.side_effect = lambda chunks: [
            {**c, "embedding": fake_vector} for c in chunks
        ]
        m_texts.return_value = [fake_vector]

        yield {"query": m_query, "chunks": m_chunks, "texts": m_texts}


@pytest.fixture
def mock_llm():
    """Patch LLM service so no real API calls are made."""
    with patch("app.services.llm_service.generate_answer", new_callable=AsyncMock) as m_answer, \
         patch("app.services.llm_service.generate_hypothetical_doc", new_callable=AsyncMock) as m_hyde:

        m_answer.return_value = {
            "answer": "This is a mocked AI answer based on the retrieved context.",
            "model_used": "gpt-4o",
            "token_count": 42,
        }
        m_hyde.return_value = "A hypothetical document about the topic."

        yield {"answer": m_answer, "hyde": m_hyde}


@pytest.fixture
def mock_reranker():
    """Patch reranker so no model is loaded during tests."""
    with patch("app.routers.query.rerank") as m_rerank:
        m_rerank.side_effect = lambda question, chunks, top_k: chunks[:top_k]
        yield m_rerank


# ── FastAPI test client ────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def async_client(
    mock_qdrant,
    mock_redis,
    mock_postgres,
    mock_embedder,
    mock_llm,
    mock_reranker,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Full integration test client with all external services mocked.
    The lifespan context is respected — startup/shutdown hooks run.
    """
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client
