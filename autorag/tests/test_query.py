"""
tests/test_query.py
====================
Integration tests for POST /query and GET /health endpoints.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


# ─────────────────────────── POST /query ──────────────────────────────────────

class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_returns_answer(self, async_client):
        """Basic query should return an answer string."""
        response = await async_client.post(
            "/query",
            json={"question": "What is retrieval-augmented generation?", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    @pytest.mark.asyncio
    async def test_query_returns_required_fields(self, async_client):
        """Response schema must include all required fields."""
        response = await async_client.post(
            "/query",
            json={"question": "Explain RAG pipeline steps.", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        required_fields = ["answer", "sources", "latency_ms", "token_count", "model_used", "question"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_query_empty_question_returns_422(self, async_client):
        """Empty question should fail validation (422)."""
        response = await async_client.post(
            "/query",
            json={"question": "", "top_k": 5},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_missing_question_returns_422(self, async_client):
        """Missing 'question' key should fail validation."""
        response = await async_client.post("/query", json={"top_k": 5})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_cache_hit_returns_fast(
        self, async_client, mock_redis, mock_embedder
    ):
        """If Redis cache has the answer, it should be returned directly."""
        mock_redis["get_cached_query"].return_value = {
            "answer": "Cached answer from Redis.",
            "sources": [],
            "latency_ms": 5,
            "token_count": 10,
            "model_used": "gpt-4o",
            "question": "What is AI?",
            "hyde_used": False,
        }
        response = await async_client.post(
            "/query", json={"question": "What is AI?", "top_k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Cached answer from Redis."
        # embedder should NOT have been called since we got a cache hit
        mock_embedder["query"].assert_not_called()

    @pytest.mark.asyncio
    async def test_query_no_results_returns_graceful_message(
        self, async_client, mock_qdrant, mock_redis, mock_embedder
    ):
        """If no search results, return a polite 'not found' message."""
        mock_qdrant["dense"].return_value = []
        mock_qdrant["hybrid"].return_value = []
        mock_redis["get_cached_query"].return_value = None

        response = await async_client.post(
            "/query",
            json={"question": "Who won the 1803 intergalactic chess tournament?", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "not find" in data["answer"].lower() or "relevant" in data["answer"].lower()

    @pytest.mark.asyncio
    async def test_query_latency_ms_is_positive(self, async_client):
        """latency_ms should always be a positive integer."""
        response = await async_client.post(
            "/query",
            json={"question": "What is embeddings?", "top_k": 5},
        )
        assert response.status_code == 200
        assert response.json()["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_query_top_k_limits_sources(
        self, async_client, mock_qdrant, mock_redis, mock_embedder, mock_llm, mock_reranker
    ):
        """The number of sources should be <= top_k."""
        mock_redis["get_cached_query"].return_value = None
        mock_qdrant["dense"].return_value = [
            {"chunk_id": f"c{i}", "score": 0.9, "text": f"chunk {i}",
             "metadata": {"doc_id": "d1", "filename": "f.pdf", "chunk_index": i,
                          "text_preview": f"chunk {i}", "file_type": "pdf", "source_url": None}}
            for i in range(10)
        ]

        response = await async_client.post(
            "/query",
            json={"question": "What is RAG?", "top_k": 3},
        )
        assert response.status_code == 200


# ─────────────────────────── GET /health ──────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_all_ok(
        self, async_client, mock_qdrant, mock_redis, mock_postgres
    ):
        """When all services are healthy, overall status should be 'ok'."""
        mock_qdrant["health"].return_value = True
        mock_redis["health"].return_value = True
        mock_postgres["health"].return_value = True

        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "qdrant" in data["services"]
        assert "redis" in data["services"]
        assert "postgres" in data["services"]

    @pytest.mark.asyncio
    async def test_health_degraded_when_redis_down(
        self, async_client, mock_qdrant, mock_redis, mock_postgres
    ):
        """If Redis fails, overall status should be 'degraded'."""
        mock_qdrant["health"].return_value = True
        mock_redis["health"].side_effect = Exception("Redis connection refused")
        mock_postgres["health"].return_value = True

        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["redis"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_health_includes_version(
        self, async_client, mock_qdrant, mock_redis, mock_postgres
    ):
        """Health response must include API version string."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert "version" in response.json()

    @pytest.mark.asyncio
    async def test_health_services_have_latency(
        self, async_client, mock_qdrant, mock_redis, mock_postgres
    ):
        """Each service entry should include latency_ms."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        for svc_name, svc_data in response.json()["services"].items():
            assert "latency_ms" in svc_data, f"{svc_name} missing latency_ms"


# ─────────────────────────── GET / (root) ─────────────────────────────────────

class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_root_returns_api_info(self, async_client):
        """Root endpoint should return API name and version."""
        response = await async_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AutoRAG"
        assert "version" in data
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_metrics_endpoint_exists(self, async_client):
        """Prometheus /metrics endpoint should be reachable."""
        response = await async_client.get("/metrics")
        assert response.status_code == 200
        assert "autorag" in response.text
