#!/usr/bin/env bash
# ──────────────────────────────────────────────
# AutoRAG — Project Structure Bootstrap Script
# Run from: /Users/vantakujagadeesh/Desktop/rag n8n/
# ──────────────────────────────────────────────
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "📂 Creating AutoRAG structure in: $BASE_DIR"

# ── 1. Create all directories ─────────────────
mkdir -p "$BASE_DIR/autorag/app/routers"
mkdir -p "$BASE_DIR/autorag/app/services"
mkdir -p "$BASE_DIR/autorag/app/models"
mkdir -p "$BASE_DIR/autorag/app/db"
mkdir -p "$BASE_DIR/autorag/app/core"
mkdir -p "$BASE_DIR/autorag/n8n-workflows"
mkdir -p "$BASE_DIR/autorag/scripts"
mkdir -p "$BASE_DIR/autorag/tests/fixtures"

# ── 2. Create app files ───────────────────────
touch "$BASE_DIR/autorag/app/__init__.py"
touch "$BASE_DIR/autorag/app/main.py"

# routers
touch "$BASE_DIR/autorag/app/routers/__init__.py"
touch "$BASE_DIR/autorag/app/routers/ingest.py"
touch "$BASE_DIR/autorag/app/routers/query.py"

# services
touch "$BASE_DIR/autorag/app/services/__init__.py"
touch "$BASE_DIR/autorag/app/services/parser.py"
touch "$BASE_DIR/autorag/app/services/chunker.py"
touch "$BASE_DIR/autorag/app/services/embedder.py"
touch "$BASE_DIR/autorag/app/services/qdrant_service.py"
touch "$BASE_DIR/autorag/app/services/redis_service.py"
touch "$BASE_DIR/autorag/app/services/reranker.py"
touch "$BASE_DIR/autorag/app/services/llm_service.py"

# models
touch "$BASE_DIR/autorag/app/models/__init__.py"
touch "$BASE_DIR/autorag/app/models/schemas.py"

# db
touch "$BASE_DIR/autorag/app/db/__init__.py"
touch "$BASE_DIR/autorag/app/db/postgres.py"

# core
touch "$BASE_DIR/autorag/app/core/__init__.py"
touch "$BASE_DIR/autorag/app/core/config.py"

# ── 3. n8n workflow placeholders ──────────────
touch "$BASE_DIR/autorag/n8n-workflows/ingestion.json"
touch "$BASE_DIR/autorag/n8n-workflows/query-handler.json"
touch "$BASE_DIR/autorag/n8n-workflows/eval-cron.json"
touch "$BASE_DIR/autorag/n8n-workflows/error-handler.json"

# ── 4. Scripts ────────────────────────────────
touch "$BASE_DIR/autorag/scripts/init_qdrant.py"
touch "$BASE_DIR/autorag/scripts/preflight_check.py"

# ── 5. Tests ──────────────────────────────────
touch "$BASE_DIR/autorag/tests/__init__.py"
touch "$BASE_DIR/autorag/tests/conftest.py"
touch "$BASE_DIR/autorag/tests/test_ingest.py"
touch "$BASE_DIR/autorag/tests/test_query.py"
# fixtures/ dir already created above (empty)

# ── 6. Root-level config files ────────────────
touch "$BASE_DIR/autorag/.env"
touch "$BASE_DIR/autorag/.env.example"
touch "$BASE_DIR/autorag/Dockerfile"
touch "$BASE_DIR/autorag/railway.toml"

# Only create these if they don't already exist
[ -f "$BASE_DIR/autorag/requirements.txt" ] || touch "$BASE_DIR/autorag/requirements.txt"
[ -f "$BASE_DIR/autorag/.gitignore" ]       || touch "$BASE_DIR/autorag/.gitignore"

echo ""
echo "✅ AutoRAG structure created successfully!"
echo ""
echo "── Verifying with tree ──"
if command -v tree &>/dev/null; then
    tree "$BASE_DIR/autorag" -a --dirsfirst
else
    find "$BASE_DIR/autorag" | sort | sed "s|$BASE_DIR/||"
fi
