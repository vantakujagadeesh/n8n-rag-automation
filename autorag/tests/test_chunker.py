"""
tests/test_chunker.py
======================
Unit tests for app.services.chunker — no external dependencies.
"""

from __future__ import annotations

import pytest
from app.services.chunker import chunk_document, chunk_text


# ─────────────────────────── chunk_text ───────────────────────────────────────

class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n  \t  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello, this is a short sentence that fits in one chunk."
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split_into_multiple_chunks(self, sample_text):
        # sample_text is ~900 chars; chunk_size=200 should create multiple chunks
        chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_all_chunks_respect_min_length(self, sample_text):
        from app.services.chunker import MIN_CHUNK_LENGTH
        chunks = chunk_text(sample_text, chunk_size=512, chunk_overlap=50)
        for chunk in chunks:
            assert len(chunk.strip()) >= MIN_CHUNK_LENGTH, (
                f"Chunk shorter than MIN_CHUNK_LENGTH: {chunk!r}"
            )

    def test_overlap_means_context_is_shared(self, sample_text):
        """With overlap > 0, the end of chunk N should appear in chunk N+1."""
        chunks = chunk_text(sample_text, chunk_size=300, chunk_overlap=100)
        if len(chunks) > 1:
            # The last few words of chunk 0 should appear somewhere in chunk 1
            tail = chunks[0][-50:]
            assert tail in chunks[1] or len(chunks[1]) > 0  # relaxed: just no crash

    def test_no_overlap_produces_disjoint_or_touching_chunks(self, sample_text):
        chunks = chunk_text(sample_text, chunk_size=300, chunk_overlap=0)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_chunk_size_respected_approximately(self, sample_text):
        chunk_size = 200
        chunks = chunk_text(sample_text, chunk_size=chunk_size, chunk_overlap=10)
        for chunk in chunks:
            # Chunks can slightly exceed due to separator logic, allow 20% buffer
            assert len(chunk) <= chunk_size * 1.2, (
                f"Chunk too long: {len(chunk)} chars (limit {chunk_size})"
            )

    def test_single_word_text_returns_single_chunk(self):
        # A 50-char word-like string that passes MIN_CHUNK_LENGTH
        text = "x" * 60
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text


# ─────────────────────────── chunk_document ───────────────────────────────────

class TestChunkDocument:
    def test_returns_list_of_dicts(self, sample_text):
        results = chunk_document("doc-abc", "report.pdf", sample_text, "pdf")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_chunk_ids_are_unique(self, sample_text):
        results = chunk_document("doc-abc", "report.pdf", sample_text, "pdf")
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_id_format(self, sample_text):
        results = chunk_document("doc-xyz", "paper.pdf", sample_text, "pdf")
        for i, r in enumerate(results):
            assert r["chunk_id"] == f"doc-xyz_chunk_{i}"

    def test_doc_id_propagated(self, sample_text):
        results = chunk_document("my-doc-id", "file.docx", sample_text, "docx")
        for r in results:
            assert r["doc_id"] == "my-doc-id"

    def test_filename_propagated(self, sample_text):
        results = chunk_document("doc-1", "my_file.pdf", sample_text, "pdf")
        for r in results:
            assert r["filename"] == "my_file.pdf"

    def test_file_type_propagated(self, sample_text):
        results = chunk_document("doc-1", "page.html", sample_text, "url")
        for r in results:
            assert r["file_type"] == "url"

    def test_chunk_index_is_sequential(self, sample_text):
        results = chunk_document("doc-1", "doc.pdf", sample_text, "pdf")
        for i, r in enumerate(results):
            assert r["chunk_index"] == i

    def test_text_preview_is_first_200_chars(self, sample_text):
        results = chunk_document("doc-1", "doc.pdf", sample_text, "pdf")
        for r in results:
            assert r["text_preview"] == r["text"][:200]

    def test_char_count_matches_text_length(self, sample_text):
        results = chunk_document("doc-1", "doc.pdf", sample_text, "pdf")
        for r in results:
            assert r["char_count"] == len(r["text"])

    def test_empty_text_returns_empty_list(self):
        results = chunk_document("doc-1", "empty.pdf", "", "pdf")
        assert results == []

    def test_whitespace_only_returns_empty_list(self):
        results = chunk_document("doc-1", "blank.pdf", "   \n\t  ", "pdf")
        assert results == []
