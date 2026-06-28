# app/models/schemas.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/app/models/schemas.py

from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Used when ingesting via URL (not file upload).
    For file uploads, FastAPI UploadFile is used directly in the router.
    """
    filename: str = Field(..., description="Original filename or a descriptive name for the URL")
    file_url: Optional[str] = Field(None, description="Public URL to fetch and ingest")
    file_type: Optional[str] = Field(None, description="pdf | docx | url — auto-detected if omitted")

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v: Optional[str]) -> Optional[str]:
        allowed = {"pdf", "docx", "url", None}
        if v not in allowed:
            raise ValueError(f"file_type must be one of: pdf, docx, url. Got: {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "company_policy.pdf",
                "file_url": "https://example.com/docs/policy.pdf",
                "file_type": "pdf",
            }
        }
    }


class IngestResponse(BaseModel):
    """Returned by POST /ingest on success."""
    doc_id: str = Field(..., description="UUID assigned to this document")
    filename: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks stored in Qdrant")
    skipped: bool = Field(False, description="True if file was a duplicate and skipped")
    latency_ms: int = Field(..., description="Total ingest pipeline latency in milliseconds")
    message: str = Field(..., description="Human-readable status message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "doc_id": "3f7a2b1c-...",
                "filename": "company_policy.pdf",
                "chunk_count": 42,
                "skipped": False,
                "latency_ms": 3240,
                "message": "Document ingested successfully: 42 chunks stored",
            }
        }
    }


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Sent by Slack n8n webhook or direct API caller to POST /query."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to answer from the knowledge base",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of source chunks to include in the response",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the company refund policy?",
                "top_k": 5,
            }
        }
    }


class ChunkMetadata(BaseModel):
    """Metadata attached to each retrieved chunk in a query response."""
    doc_id: str = Field(..., description="ID of the parent document")
    filename: str = Field(..., description="Original filename of the source document")
    chunk_index: int = Field(..., description="Zero-based position of this chunk in the document")
    text_preview: str = Field(..., description="First 200 chars of chunk text for display")
    file_type: Optional[str] = Field(None, description="pdf | docx | url")
    source_url: Optional[str] = Field(None, description="Original URL if ingested from web")


class SearchResult(BaseModel):
    """A single retrieved and reranked chunk returned from the vector store."""
    chunk_id: str = Field(..., description="Unique ID of this chunk in Qdrant")
    score: float = Field(..., description="Reranker confidence score (0.0 – 1.0)")
    text: str = Field(..., description="Full text of the retrieved chunk")
    metadata: ChunkMetadata = Field(..., description="Document metadata for this chunk")


class QueryResponse(BaseModel):
    """Returned by POST /query. Contains answer + cited sources."""
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved chunks")
    sources: list[SearchResult] = Field(
        default_factory=list,
        description="Top-K reranked source chunks used to generate the answer",
    )
    latency_ms: int = Field(..., description="End-to-end query pipeline latency in milliseconds")
    token_count: int = Field(..., description="Total tokens used by the LLM call")
    model_used: str = Field(..., description="LLM model that generated the answer")
    question: str = Field(..., description="Echo of the original question")
    hyde_used: bool = Field(False, description="Whether HyDE expansion was applied")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The refund policy allows returns within 30 days...",
                "sources": [],
                "latency_ms": 1820,
                "token_count": 540,
                "model_used": "gpt-4o",
                "question": "What is the company refund policy?",
                "hyde_used": True,
            }
        }
    }


# ── Health ────────────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    """Status of a single downstream service."""
    status: str = Field(..., description="ok | error")
    latency_ms: Optional[int] = Field(None, description="Ping latency in milliseconds")
    detail: Optional[str] = Field(None, description="Error message if status is error")


class HealthResponse(BaseModel):
    """Returned by GET /health — aggregated status of all services."""
    status: str = Field(..., description="ok | degraded | error")
    version: str = Field(..., description="App version")
    services: dict[str, ServiceStatus] = Field(
        ..., description="Per-service health: qdrant, redis, postgres"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "services": {
                    "qdrant":   {"status": "ok", "latency_ms": 42},
                    "redis":    {"status": "ok", "latency_ms": 3},
                    "postgres": {"status": "ok", "latency_ms": 8},
                },
                "timestamp": "2025-01-01T00:00:00Z",
            }
        }
    }


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx/5xx responses."""
    error: str = Field(..., description="Short error code e.g. 'duplicate_document'")
    detail: str = Field(..., description="Human-readable explanation of what went wrong")
    doc_id: Optional[str] = Field(None, description="Relevant doc_id if applicable")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "duplicate_document",
                "detail": "This file was already ingested (doc_id: abc-123)",
                "doc_id": "abc-123",
            }
        }
    }


# ── Document list (bonus — used by GET /documents) ───────────────────────────

class DocumentSummary(BaseModel):
    """Lightweight document info for listing endpoints."""
    id: str
    filename: str
    chunk_count: int
    file_type: Optional[str]
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int
    limit: int
    offset: int
