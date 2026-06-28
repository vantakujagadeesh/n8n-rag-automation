# AutoRAG 🚀

> **Production-grade RAG pipeline** — ingest PDFs, DOCX, and URLs into a hybrid vector search system powered by OpenAI, Qdrant, Redis, and PostgreSQL. Answer questions with cited sources via GPT-4o.

---

## Architecture

```
Upload PDF/DOCX/URL
       ↓
   Parse (PyMuPDF / python-docx / Playwright)
       ↓
   Chunk (512 tokens, 50 overlap)
       ↓
   SHA-256 Dedup → Redis
       ↓
   Embed (text-embedding-3-large, 3072 dims)
       ↓
   Upsert → Qdrant  +  log → PostgreSQL
       
Query:
   HyDE expansion → Embed → Hybrid Search (dense + BM25)
       ↓ RRF fusion
   Cross-encoder Rerank (top-20 → top-5)
       ↓
   GPT-4o answer with citations → Redis cache → logged
```

## Stack

| Layer | Tech |
|-------|------|
| API | FastAPI + Uvicorn |
| Vector DB | Qdrant Cloud |
| Cache | Redis |
| Metadata DB | PostgreSQL (asyncpg) |
| Embeddings | OpenAI text-embedding-3-large |
| LLM | GPT-4o (fallback: Claude Sonnet) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Workflows | n8n |
| Storage | AWS S3 (optional) |
| Deploy | Railway / Docker |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo>
cd autorag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your real API keys
nano .env
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

- **Swagger UI:** http://localhost:8000/docs  
- **Dashboard:** Open `dashboard.html` in browser  
- **Health check:** http://localhost:8000/health

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `ANTHROPIC_API_KEY` | ✅ | Anthropic fallback key |
| `QDRANT_URL` | ✅ | Qdrant Cloud URL |
| `QDRANT_API_KEY` | ✅ | Qdrant API key |
| `REDIS_URL` | ✅ | Redis connection URL |
| `DATABASE_URL` | ✅ | PostgreSQL URL |
| `QDRANT_COLLECTION` | ❌ | Default: `documents` |
| `EMBEDDING_MODEL` | ❌ | Default: `text-embedding-3-large` |
| `EMBEDDING_DIMS` | ❌ | Default: `3072` |
| `PRIMARY_LLM` | ❌ | Default: `gpt-4o` |
| `CHUNK_SIZE` | ❌ | Default: `512` |
| `CHUNK_OVERLAP` | ❌ | Default: `50` |
| `RETRIEVAL_TOP_K` | ❌ | Default: `20` |
| `RERANK_TOP_K` | ❌ | Default: `5` |
| `HYDE_ENABLED` | ❌ | Default: `true` |
| `S3_BUCKET` | ❌ | Optional S3 storage |
| `SLACK_BOT_TOKEN` | ❌ | Optional Slack alerts |

---

## API Endpoints

### POST /ingest
Upload PDF/DOCX or ingest a URL.
```bash
# File upload
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"

# URL
curl -X POST http://localhost:8000/ingest \
  -F "file_url=https://example.com/doc.pdf"
```

### POST /query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "top_k": 5}'
```

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /documents
```bash
curl "http://localhost:8000/documents?limit=20&offset=0"
```

### DELETE /documents/{doc_id}
```bash
curl -X DELETE http://localhost:8000/documents/your-doc-id
```

---

## Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Set all environment variables in the Railway dashboard.

## Deploy with Docker

```bash
docker build -t autorag .
docker run -p 8000:8000 --env-file .env autorag
```

---

## Project Structure

```
autorag/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── core/config.py       # Pydantic settings
│   ├── db/postgres.py       # AsyncPG pool + queries
│   ├── models/schemas.py    # Pydantic models
│   ├── routers/
│   │   ├── ingest.py        # POST /ingest, GET /documents
│   │   └── query.py         # POST /query, GET /health
│   └── services/
│       ├── parser.py        # PDF/DOCX/URL extraction
│       ├── chunker.py       # Text splitting
│       ├── embedder.py      # OpenAI embeddings
│       ├── qdrant_service.py# Vector store
│       ├── redis_service.py # Cache + dedup
│       ├── reranker.py      # Cross-encoder rerank
│       └── llm_service.py   # GPT-4o answer generation
├── n8n-workflows/           # n8n automation workflows
├── scripts/
│   ├── init_qdrant.py       # Create Qdrant collection
│   └── preflight_check.py  # Validate all services
├── dashboard.html           # Web dashboard preview
└── Dockerfile
```

---

## License

MIT
