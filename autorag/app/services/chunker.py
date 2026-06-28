# app/services/chunker.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/services/chunker.py
#
# Text chunking service using LangChain's RecursiveCharacterTextSplitter.
# Produces chunk dicts ready for embedding and Qdrant upsert.
# ──────────────────────────────────────────────────────────────────────────────

import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Minimum chunk length to keep (shorter chunks add noise, not signal)
MIN_CHUNK_LENGTH: int = 50

# Ordered separators: prefer splitting at paragraph > line > sentence > word > char
SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Split a document's text into overlapping chunks using recursive character splitting.

    The splitter tries the most semantically meaningful separators first
    (paragraph breaks → newlines → sentences → words → characters),
    falling back only when a chunk would exceed the target size.

    Args:
        text:          Raw document text to split.
        chunk_size:    Target chunk size in characters (default from config: 512).
        chunk_overlap: Overlap between consecutive chunks (default from config: 50).

    Returns:
        List of chunk strings, each between MIN_CHUNK_LENGTH and ~chunk_size chars.
        Empty list if input text is empty or below minimum threshold.
    """
    if not text or not text.strip():
        logger.warning("chunk_text called with empty text — returning empty list")
        return []

    input_chars = len(text)
    logger.info(
        f"Chunking {input_chars:,} chars "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        strip_whitespace=True,
    )

    raw_chunks: list[str] = splitter.split_text(text)

    # Filter out fragments that are too short to be meaningful
    chunks = [
        chunk for chunk in raw_chunks
        if len(chunk.strip()) >= MIN_CHUNK_LENGTH
    ]

    filtered_count = len(raw_chunks) - len(chunks)
    if filtered_count > 0:
        logger.debug(
            f"Filtered out {filtered_count} chunks below "
            f"{MIN_CHUNK_LENGTH} char threshold"
        )

    logger.info(
        f"Chunking complete: {input_chars:,} chars → {len(chunks)} chunks "
        f"(avg {input_chars // max(len(chunks), 1)} chars/chunk)"
    )

    return chunks


def chunk_document(
    doc_id: str,
    filename: str,
    text: str,
    file_type: str,
) -> list[dict]:
    """
    Chunk a document and return structured dicts ready for embedding + Qdrant upsert.

    Each dict contains all metadata needed downstream:
      - chunk_id:     unique identifier "{doc_id}_chunk_{index}"
      - doc_id:       parent document UUID
      - filename:     original source filename
      - file_type:    "pdf" | "docx" | "url"
      - chunk_index:  zero-based position within the document
      - text:         full chunk text (for embedding)
      - text_preview: first 200 chars (for API responses / Slack previews)
      - char_count:   length of the chunk text

    Args:
        doc_id:    UUID string assigned to this document at ingest time.
        filename:  Original filename (e.g. "policy.pdf").
        text:      Full extracted document text (output of parser).
        file_type: "pdf" | "docx" | "url".

    Returns:
        List of chunk dicts. Empty list if text produces no valid chunks.
    """
    logger.info(
        f"Chunking document: doc_id={doc_id}, "
        f"filename='{filename}', type={file_type}, "
        f"text_length={len(text):,} chars"
    )

    chunks = chunk_text(text)

    if not chunks:
        logger.warning(
            f"Document '{filename}' (id={doc_id}) produced zero chunks "
            f"from {len(text):,} chars of text"
        )
        return []

    chunk_dicts: list[dict] = []
    for i, chunk in enumerate(chunks):
        chunk_dicts.append(
            {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
                "text": chunk,
                "text_preview": chunk[:200],
                "char_count": len(chunk),
            }
        )

    logger.info(
        f"Document chunked: '{filename}' → {len(chunk_dicts)} chunks "
        f"(ids: {chunk_dicts[0]['chunk_id']} … {chunk_dicts[-1]['chunk_id']})"
    )

    return chunk_dicts
