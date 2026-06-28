"""
tests/test_embedder.py
=======================
Unit tests for app.services.embedder — mocks the OpenAI client.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────── embed_texts ──────────────────────────────────────

class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        from app.services.embedder import embed_texts
        result = await embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_text_returns_single_vector(self, mock_openai_client):
        from app.services.embedder import embed_texts
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(index=0, embedding=[0.1] * 3072)]
        )
        result = await embed_texts(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 3072

    @pytest.mark.asyncio
    async def test_multiple_texts_returns_matching_count(self, mock_openai_client):
        from app.services.embedder import embed_texts
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(index=0, embedding=[0.1] * 3072),
                MagicMock(index=1, embedding=[0.2] * 3072),
                MagicMock(index=2, embedding=[0.3] * 3072),
            ]
        )
        result = await embed_texts(["a", "b", "c"])
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embeddings_maintain_order(self, mock_openai_client):
        """Embeddings must match input order (API sorts by index)."""
        from app.services.embedder import embed_texts
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(index=1, embedding=[0.9] * 3072),
                MagicMock(index=0, embedding=[0.1] * 3072),
            ]
        )
        result = await embed_texts(["first", "second"])
        # After sorting by index, first item should have 0.1 embedding
        assert result[0][0] == pytest.approx(0.1)
        assert result[1][0] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_batching_makes_multiple_api_calls(self, mock_openai_client):
        """100+ texts should be split into multiple API batches."""
        from app.services.embedder import embed_texts, _BATCH_SIZE
        total_texts = _BATCH_SIZE + 10  # Force 2 batches

        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(index=i, embedding=[float(i)] * 3072) for i in range(total_texts)]
        )
        # Override per-batch mock separately
        batch1_response = MagicMock(
            data=[MagicMock(index=i, embedding=[0.1] * 3072) for i in range(_BATCH_SIZE)]
        )
        batch2_response = MagicMock(
            data=[MagicMock(index=i, embedding=[0.2] * 3072) for i in range(10)]
        )
        mock_openai_client.embeddings.create.side_effect = [batch1_response, batch2_response]

        result = await embed_texts(["text"] * total_texts)
        assert len(result) == total_texts
        assert mock_openai_client.embeddings.create.call_count == 2


# ─────────────────────────── embed_query ──────────────────────────────────────

class TestEmbedQuery:
    @pytest.mark.asyncio
    async def test_returns_single_vector(self, mock_openai_client):
        from app.services.embedder import embed_query
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(index=0, embedding=[0.5] * 3072)]
        )
        result = await embed_query("What is machine learning?")
        assert len(result) == 3072
        assert all(v == pytest.approx(0.5) for v in result)

    @pytest.mark.asyncio
    async def test_empty_string_raises_value_error(self):
        from app.services.embedder import embed_query
        with pytest.raises(ValueError, match="non-empty string"):
            await embed_query("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises_value_error(self):
        from app.services.embedder import embed_query
        with pytest.raises(ValueError, match="non-empty string"):
            await embed_query("   ")


# ─────────────────────────── embed_chunks ─────────────────────────────────────

class TestEmbedChunks:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        from app.services.embedder import embed_chunks
        result = await embed_chunks([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embedding_attached_to_each_chunk(
        self, mock_openai_client, sample_chunk_dicts
    ):
        from app.services.embedder import embed_chunks
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(index=0, embedding=[0.1] * 3072),
                MagicMock(index=1, embedding=[0.2] * 3072),
            ]
        )
        result = await embed_chunks(sample_chunk_dicts)
        assert len(result) == len(sample_chunk_dicts)
        for chunk in result:
            assert "embedding" in chunk
            assert len(chunk["embedding"]) == 3072

    @pytest.mark.asyncio
    async def test_original_fields_preserved(
        self, mock_openai_client, sample_chunk_dicts
    ):
        from app.services.embedder import embed_chunks
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(index=0, embedding=[0.1] * 3072),
                MagicMock(index=1, embedding=[0.2] * 3072),
            ]
        )
        result = await embed_chunks(sample_chunk_dicts)
        for original, enriched in zip(sample_chunk_dicts, result):
            assert enriched["chunk_id"] == original["chunk_id"]
            assert enriched["text"] == original["text"]
            assert enriched["doc_id"] == original["doc_id"]

    @pytest.mark.asyncio
    async def test_does_not_mutate_input(
        self, mock_openai_client, sample_chunk_dicts
    ):
        """embed_chunks must return new dicts, not modify the originals."""
        from app.services.embedder import embed_chunks
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(index=0, embedding=[0.1] * 3072),
                MagicMock(index=1, embedding=[0.2] * 3072),
            ]
        )
        original_keys = set(sample_chunk_dicts[0].keys())
        await embed_chunks(sample_chunk_dicts)
        assert set(sample_chunk_dicts[0].keys()) == original_keys
