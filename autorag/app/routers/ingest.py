# app/routers/ingest.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/routers/ingest.py
"""
AutoRAG — Ingest Router
POST   /ingest             — Ingest PDF/DOCX file or URL
GET    /documents          — List ingested documents (paginated)
DELETE /documents/{doc_id} — Delete a document from all stores
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from typing import Annotated, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.db import postgres
from app.models.schemas import DocumentListResponse, DocumentSummary, ErrorResponse, IngestResponse
from app.services import chunker, embedder, qdrant_service, redis_service
from app.services.parser import detect_file_type, parse_file, parse_url

router = APIRouter(tags=["ingest"])
logger = logging.getLogger(__name__)
settings = get_settings()

_MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB


async def _upload_to_s3(file_bytes: bytes, doc_id: str, filename: str) -> Optional[str]:
    """Upload raw bytes to S3. Returns key on success, None if not configured or fails."""
    if not settings.is_s3_configured:
        return None
    key = f"documents/{doc_id}/{filename}"

    def _sync_upload() -> str:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        s3.upload_fileobj(io.BytesIO(file_bytes), settings.s3_bucket, key,
                          ExtraArgs={"ContentType": "application/octet-stream"})
        return key

    try:
        s3_key = await asyncio.to_thread(_sync_upload)
        logger.info("S3 upload OK: s3://%s/%s", settings.s3_bucket, s3_key)
        return s3_key
    except (BotoCoreError, ClientError) as exc:
        logger.warning("S3 upload failed (non-fatal): %s", exc)
        return None


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a document (file upload or URL)",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def ingest_document(
    file: Annotated[Optional[UploadFile], File(description="PDF or DOCX file")] = None,
    file_url: Annotated[Optional[str], Form(description="Public URL to ingest")] = None,
    filename: Annotated[Optional[str], Form(description="Filename override for URL")] = None,
) -> IngestResponse:
    """
    Ingest a document via file upload (PDF/DOCX) or URL.
    Runs: dedup → parse → chunk → embed → Qdrant → Postgres → Redis → S3.
    """
    start_time = time.time()

    if file is None and not file_url:
        raise HTTPException(status_code=400, detail="Provide a file upload or file_url.")

    source_url: Optional[str] = None
    file_bytes: bytes
    effective_filename: str

    if file is not None:
        effective_filename = file.filename or "upload.bin"
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(file_bytes) > _MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="File exceeds 50 MB limit.")
        try:
            detect_file_type(effective_filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        source_url = file_url
        effective_filename = filename or (file_url.split("/")[-1].split("?")[0] or "page.html")
        file_bytes = source_url.encode("utf-8")

    logger.info("Ingest: filename='%s' source_url=%s", effective_filename, source_url)

    # ── Dedup ──────────────────────────────────────────────────────────────────
    file_hash = redis_service.get_file_hash(file_bytes)
    if await redis_service.is_duplicate(file_hash):
        logger.info("Duplicate — skipping hash=%s…", file_hash[:12])
        latency_ms = int((time.time() - start_time) * 1000)
        return IngestResponse(
            doc_id="duplicate",
            filename=effective_filename,
            chunk_count=0,
            skipped=True,
            latency_ms=latency_ms,
            message=f"Already ingested (hash: {file_hash[:16]}…). Skipped.",
        )

    doc_id = str(uuid.uuid4())
    logger.info("New doc_id=%s", doc_id)

    try:
        # ── File type ──────────────────────────────────────────────────────────
        file_type = "url" if source_url else detect_file_type(effective_filename)

        # ── Parse ──────────────────────────────────────────────────────────────
        if file_type == "url":
            raw_text = await parse_url(source_url)
        else:
            raw_text = await asyncio.to_thread(parse_file, effective_filename, file_bytes)

        if not raw_text or not raw_text.strip():
            raise ValueError("Parser returned empty text.")
        logger.info("Parsed %d chars", len(raw_text))

        # ── Chunk ──────────────────────────────────────────────────────────────
        chunk_dicts = await asyncio.to_thread(
            chunker.chunk_document, doc_id, effective_filename, raw_text, file_type
        )
        if not chunk_dicts:
            raise ValueError("Chunking produced zero chunks.")
        logger.info("Chunked: %d chunks", len(chunk_dicts))

        # ── Embed ──────────────────────────────────────────────────────────────
        chunks_with_embeddings = await embedder.embed_chunks(chunk_dicts)
        logger.info("Embedded: %d chunks", len(chunks_with_embeddings))

        # ── Qdrant upsert ──────────────────────────────────────────────────────
        upserted_count = await qdrant_service.upsert_chunks(chunks_with_embeddings)
        logger.info("Qdrant: %d points upserted", upserted_count)

        # ── PostgreSQL ─────────────────────────────────────────────────────────
        await postgres.save_document(
            doc_id=doc_id,
            filename=effective_filename,
            file_hash=file_hash,
            chunk_count=upserted_count,
            file_type=file_type,
            source_url=source_url,
        )

        # ── Redis mark ─────────────────────────────────────────────────────────
        await redis_service.mark_ingested(file_hash, doc_id)

        # ── S3 upload (optional, files only) ──────────────────────────────────
        if file is not None:
            await _upload_to_s3(file_bytes, doc_id, effective_filename)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("Ingest OK: doc_id=%s chunks=%d latency=%dms", doc_id, upserted_count, latency_ms)
        return IngestResponse(
            doc_id=doc_id,
            filename=effective_filename,
            chunk_count=upserted_count,
            skipped=False,
            latency_ms=latency_ms,
            message=f"Document ingested successfully: {upserted_count} chunks stored.",
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.exception("Ingest FAILED: doc_id=%s latency=%dms", doc_id, latency_ms)
        raise HTTPException(status_code=500, detail=f"Ingest pipeline error: {exc}") from exc


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List ingested documents",
)
async def list_documents(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """Returns a paginated list of all documents in the knowledge base."""
    try:
        rows = await postgres.list_documents(limit=limit, offset=offset)
        documents = [
            DocumentSummary(
                id=row["id"],
                filename=row["filename"],
                chunk_count=row["chunk_count"],
                file_type=row.get("file_type"),
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return DocumentListResponse(documents=documents, total=len(documents), limit=limit, offset=offset)
    except Exception as exc:
        logger.exception("list_documents failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}") from exc


@router.delete(
    "/documents/{doc_id}",
    summary="Delete a document from Qdrant",
)
async def delete_document(doc_id: str) -> JSONResponse:
    """Removes all Qdrant vectors for this doc_id."""
    try:
        deleted_count = await qdrant_service.delete_document(doc_id)
        logger.info("Deleted doc_id=%s: %d points removed", doc_id, deleted_count)
        return JSONResponse(
            status_code=200,
            content={"deleted": True, "doc_id": doc_id, "points_removed": deleted_count},
        )
    except Exception as exc:
        logger.exception("delete_document failed: doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc
