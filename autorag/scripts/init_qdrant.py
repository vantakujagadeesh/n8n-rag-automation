#!/usr/bin/env python3
# scripts/init_qdrant.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/scripts/init_qdrant.py
"""
AutoRAG — Qdrant Initialisation Script
=======================================
Run ONCE before the first deploy (or any time the collection is wiped).

Usage:
    cd /Users/vantakujagadeesh/Desktop/rag\ n8n/autorag
    python scripts/init_qdrant.py

Exit codes:
    0 — all steps succeeded
    1 — at least one step failed
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Load .env BEFORE importing app modules ────────────────────────────────────
# Locate the .env file relative to this script (../  relative to scripts/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_env_path, override=True)
    print(f"📄  Loaded environment from {_env_path}")
else:
    print(f"⚠️   No .env found at {_env_path} — using system environment variables")

# Add project root to sys.path so `from app...` imports work
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from qdrant_client import QdrantClient, models  # sync client for scripts
from qdrant_client.http.exceptions import UnexpectedResponse

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION", "documents")
EMBEDDING_DIMS: int = int(os.environ.get("EMBEDDING_DIMS", "3072"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(step: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  ✅  {step}{suffix}")


def _fail(step: str, error: str) -> None:
    print(f"  ❌  {step}  FAILED: {error}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Runs all Qdrant initialisation steps.
    Returns 0 on full success, 1 on any failure.
    """
    print()
    print("=" * 60)
    print("  AutoRAG — Qdrant Initialisation")
    print(f"  URL        : {QDRANT_URL}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Dims       : {EMBEDDING_DIMS}")
    print("=" * 60)
    print()

    failures = 0

    # ── Step 1: Connect ───────────────────────────────────────────────────────
    print("[1/5] Connecting to Qdrant…")
    t0 = time.perf_counter()
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
        # Validate connection
        collections = client.get_collections()
        elapsed = int((time.perf_counter() - t0) * 1000)
        existing_names = {c.name for c in collections.collections}
        _ok("Qdrant connection", f"{elapsed}ms — {len(existing_names)} existing collection(s)")
    except Exception as exc:
        _fail("Qdrant connection", str(exc))
        print("\n⛔  Cannot proceed without Qdrant. Aborting.\n")
        return 1

    # ── Step 2: Create collection ─────────────────────────────────────────────
    print(f"\n[2/5] Creating collection '{COLLECTION_NAME}'…")
    t0 = time.perf_counter()
    try:
        if COLLECTION_NAME in existing_names:
            elapsed = int((time.perf_counter() - t0) * 1000)
            _ok(f"Collection '{COLLECTION_NAME}'", "already exists — skipped")
        else:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=EMBEDDING_DIMS,
                        distance=models.Distance.COSINE,
                        on_disk=False,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                    ),
                },
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            )
            elapsed = int((time.perf_counter() - t0) * 1000)
            _ok(
                f"Collection '{COLLECTION_NAME}' created",
                f"dense={EMBEDDING_DIMS}d Cosine + BM25 sparse | {elapsed}ms",
            )
    except UnexpectedResponse as exc:
        if exc.status_code == 409:
            _ok(f"Collection '{COLLECTION_NAME}'", "already exists (409 — idempotent)")
        else:
            _fail(f"Create collection '{COLLECTION_NAME}'", str(exc))
            failures += 1
    except Exception as exc:
        _fail(f"Create collection '{COLLECTION_NAME}'", str(exc))
        failures += 1

    # ── Step 3: Payload index — doc_id ────────────────────────────────────────
    print(f"\n[3/5] Creating payload index: doc_id…")
    t0 = time.perf_counter()
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        _ok("Index: doc_id (KEYWORD)", f"{elapsed}ms")
    except UnexpectedResponse as exc:
        if exc.status_code in (400, 409) and "already exists" in str(exc).lower():
            _ok("Index: doc_id", "already exists — skipped")
        else:
            _fail("Index: doc_id", str(exc))
            failures += 1
    except Exception as exc:
        _fail("Index: doc_id", str(exc))
        failures += 1

    # ── Step 4: Payload index — filename ──────────────────────────────────────
    print(f"\n[4/5] Creating payload index: filename…")
    t0 = time.perf_counter()
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="filename",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        _ok("Index: filename (KEYWORD)", f"{elapsed}ms")
    except UnexpectedResponse as exc:
        if exc.status_code in (400, 409) and "already exists" in str(exc).lower():
            _ok("Index: filename", "already exists — skipped")
        else:
            _fail("Index: filename", str(exc))
            failures += 1
    except Exception as exc:
        _fail("Index: filename", str(exc))
        failures += 1

    # ── Step 5: Payload index — file_type ─────────────────────────────────────
    print(f"\n[5/5] Creating payload index: file_type…")
    t0 = time.perf_counter()
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="file_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        _ok("Index: file_type (KEYWORD)", f"{elapsed}ms")
    except UnexpectedResponse as exc:
        if exc.status_code in (400, 409) and "already exists" in str(exc).lower():
            _ok("Index: file_type", "already exists — skipped")
        else:
            _fail("Index: file_type", str(exc))
            failures += 1
    except Exception as exc:
        _fail("Index: file_type", str(exc))
        failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if failures == 0:
        # Verify final collection state
        info = client.get_collection(COLLECTION_NAME)
        point_count = info.points_count or 0
        print(f"  🚀  Qdrant init COMPLETE — collection '{COLLECTION_NAME}'")
        print(f"      Points: {point_count} | Status: {info.status}")
        print("=" * 60)
        print()
        return 0
    else:
        print(f"  ⛔  {failures} step(s) FAILED — review errors above before deploying.")
        print("=" * 60)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
