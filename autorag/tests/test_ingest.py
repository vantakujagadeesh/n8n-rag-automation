"""
tests/test_ingest.py
=====================
Integration tests for POST /ingest, GET /documents, DELETE /documents/{id}.

All external dependencies (Qdrant, Redis, PostgreSQL, OpenAI, S3) are mocked
via the conftest fixtures so no real services are required.
"""

from __future__ import annotations

import io
import pytest
from unittest.mock import patch, AsyncMock


# ─────────────────────────── POST /ingest ─────────────────────────────────────

class TestIngestEndpoint:
    @pytest.mark.asyncio
    async def test_ingest_valid_pdf(self, async_client, mock_redis, mock_qdrant, mock_postgres):
        """A valid PDF upload should return 200 with doc_id and chunk_count."""
        with patch("app.routers.ingest.parse_file", return_value="Sample document text content for testing."), \
             patch("app.routers.ingest.detect_file_type", return_value="pdf"), \
             patch("app.routers.ingest._upload_to_s3", new_callable=AsyncMock, return_value=None):

            file_content = b"%PDF-1.4 fake pdf content for testing purposes"
            response = await async_client.post(
                "/ingest",
                files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert "doc_id" in data
        assert data["doc_id"] != "duplicate"
        assert data["chunk_count"] >= 0
        assert data["skipped"] is False

    @pytest.mark.asyncio
    async def test_ingest_no_file_or_url_returns_400(self, async_client):
        """Submitting nothing should return 400."""
        response = await async_client.post("/ingest")
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_duplicate_is_skipped(
        self, async_client, mock_redis, mock_qdrant, mock_postgres
    ):
        """If the file hash already exists in Redis, return skipped=True."""
        mock_redis["is_duplicate"].return_value = True

        with patch("app.routers.ingest.detect_file_type", return_value="pdf"):
            file_content = b"%PDF-1.4 already ingested content"
            response = await async_client.post(
                "/ingest",
                files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["skipped"] is True
        assert data["doc_id"] == "duplicate"

    @pytest.mark.asyncio
    async def test_ingest_empty_file_returns_400(self, async_client, mock_redis):
        """Empty file upload should return 400."""
        with patch("app.routers.ingest.detect_file_type", return_value="pdf"):
            response = await async_client.post(
                "/ingest",
                files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
            )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_ingest_url(self, async_client, mock_redis, mock_qdrant, mock_postgres):
        """Ingesting via URL should succeed."""
        with patch("app.services.parser.parse_url", new_callable=AsyncMock,
                   return_value="Content from web page for testing."):
            response = await async_client.post(
                "/ingest",
                data={"file_url": "https://example.com/document.html"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["skipped"] is False

    @pytest.mark.asyncio
    async def test_ingest_docx_file(self, async_client, mock_redis, mock_qdrant, mock_postgres):
        """DOCX file upload should be accepted."""
        with patch("app.routers.ingest.parse_file", return_value="Word document text content."), \
             patch("app.routers.ingest.detect_file_type", return_value="docx"), \
             patch("app.routers.ingest._upload_to_s3", new_callable=AsyncMock, return_value=None):

            response = await async_client.post(
                "/ingest",
                files={"file": ("report.docx", io.BytesIO(b"fake docx bytes"), "application/octet-stream")},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_ingest_response_includes_latency(
        self, async_client, mock_redis, mock_qdrant, mock_postgres
    ):
        """Response must contain latency_ms field."""
        with patch("app.routers.ingest.parse_file", return_value="Some content."), \
             patch("app.routers.ingest.detect_file_type", return_value="pdf"), \
             patch("app.routers.ingest._upload_to_s3", new_callable=AsyncMock, return_value=None):

            response = await async_client.post(
                "/ingest",
                files={"file": ("doc.pdf", io.BytesIO(b"%PDF test"), "application/pdf")},
            )
        assert "latency_ms" in response.json()


# ─────────────────────────── GET /documents ───────────────────────────────────

class TestListDocuments:
    @pytest.mark.asyncio
    async def test_list_documents_empty(self, async_client, mock_postgres):
        """Empty knowledge base should return empty list."""
        mock_postgres["list"].return_value = []
        response = await async_client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_documents_with_results(self, async_client, mock_postgres):
        """Should return document summaries from PostgreSQL."""
        from datetime import datetime, timezone
        mock_postgres["list"].return_value = [
            {
                "id": "doc-1",
                "filename": "report.pdf",
                "chunk_count": 12,
                "file_type": "pdf",
                "created_at": datetime.now(timezone.utc),
            }
        ]
        response = await async_client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_list_documents_pagination(self, async_client, mock_postgres):
        """Limit and offset parameters should be accepted."""
        mock_postgres["list"].return_value = []
        response = await async_client.get("/documents?limit=10&offset=5")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 5

    @pytest.mark.asyncio
    async def test_list_documents_invalid_limit(self, async_client):
        """limit=0 should return 422 validation error."""
        response = await async_client.get("/documents?limit=0")
        assert response.status_code == 422


# ─────────────────────────── DELETE /documents/{id} ───────────────────────────

class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_existing_document(self, async_client, mock_qdrant):
        """Deleting a document should return 200 with deleted=True."""
        mock_qdrant["delete"].return_value = 5
        response = await async_client.delete("/documents/some-doc-id")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["doc_id"] == "some-doc-id"
        assert data["points_removed"] == 5

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, async_client, mock_qdrant):
        """Deleting a non-existent doc should still return 200 (0 points removed)."""
        mock_qdrant["delete"].return_value = 0
        response = await async_client.delete("/documents/ghost-id")
        assert response.status_code == 200
        data = response.json()
        assert data["points_removed"] == 0
