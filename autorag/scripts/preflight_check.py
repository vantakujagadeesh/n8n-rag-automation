#!/usr/bin/env python3
# scripts/preflight_check.py
# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/scripts/preflight_check.py
"""
AutoRAG — Pre-flight Service Check
====================================
Verifies connectivity and credentials for every service before deploy.

Usage:
    cd /Users/vantakujagadeesh/Desktop/rag\ n8n/autorag
    python scripts/preflight_check.py

Services tested (in order):
    1. OpenAI       — embed one string, verify 3072 dims          [REQUIRED]
    2. Anthropic    — send "hi", verify non-empty response         [REQUIRED]
    3. Qdrant       — get_collections(), verify connection         [REQUIRED]
    4. Redis        — PING, SET, GET, DELETE round-trip            [REQUIRED]
    5. PostgreSQL   — SELECT 1 via asyncpg                         [REQUIRED]
    6. AWS S3       — list_buckets() (skipped if not configured)   [OPTIONAL]
    7. Slack        — auth_test()    (skipped if not configured)   [OPTIONAL]

Exit codes:
    0 — all required services pass (optional services may warn)
    1 — one or more required services failed
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# ── Load .env before any imports ──────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_env_path, override=True)

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# ── ANSI color codes (TTY-safe) ───────────────────────────────────────────────
_IS_TTY = sys.stdout.isatty()

def _green(s: str) -> str:  return f"\033[32m{s}\033[0m" if _IS_TTY else s
def _red(s: str) -> str:    return f"\033[31m{s}\033[0m" if _IS_TTY else s
def _yellow(s: str) -> str: return f"\033[33m{s}\033[0m" if _IS_TTY else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m"  if _IS_TTY else s


# ── Result tracking ───────────────────────────────────────────────────────────

class _Result:
    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required
        self.passed: bool | None = None
        self.skipped: bool = False
        self.detail: str = ""
        self.latency_ms: int = 0

    def ok(self, detail: str, latency_ms: int) -> None:
        self.passed = True
        self.detail = detail
        self.latency_ms = latency_ms
        label = f"{'✅'} {_green(f'{self.name:<14}')} OK"
        print(f"  {label}  ({detail})  [{latency_ms}ms]")

    def fail(self, error: str, latency_ms: int) -> None:
        self.passed = False
        self.detail = error
        self.latency_ms = latency_ms
        label = f"{'❌'} {_red(f'{self.name:<14}')} FAIL"
        print(f"  {label}: {error}  [{latency_ms}ms]")

    def skip(self, reason: str) -> None:
        self.skipped = True
        self.passed = True   # optional — don't count as failure
        self.detail = reason
        label = f"{'⚠️ '} {_yellow(f'{self.name:<14}')} SKIP"
        print(f"  {label}: {reason}")


# ── Individual checks ─────────────────────────────────────────────────────────

async def check_openai() -> _Result:
    """Embed one string and verify output dimensions == 3072."""
    result = _Result("OpenAI", required=True)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
    expected_dims = int(os.environ.get("EMBEDDING_DIMS", "3072"))

    if not api_key or api_key.startswith("sk-xxxx"):
        result.fail("OPENAI_API_KEY not set or is placeholder", 0)
        return result

    t0 = time.perf_counter()
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key, timeout=30)
        response = await client.embeddings.create(
            model=model,
            input=["AutoRAG preflight test"],
        )
        dims = len(response.data[0].embedding)
        elapsed = int((time.perf_counter() - t0) * 1000)
        if dims != expected_dims:
            result.fail(f"Unexpected embedding dims: got {dims}, expected {expected_dims}", elapsed)
        else:
            result.ok(f"model={model} dims={dims}", elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result.fail(str(exc), elapsed)
    return result


async def check_anthropic() -> _Result:
    """Send a minimal message and verify we get a response string back."""
    result = _Result("Anthropic", required=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("FALLBACK_LLM", "claude-sonnet-4-6")

    if not api_key or api_key.startswith("sk-ant-xxxx"):
        result.fail("ANTHROPIC_API_KEY not set or is placeholder", 0)
        return result

    t0 = time.perf_counter()
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=30)
        response = await client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        elapsed = int((time.perf_counter() - t0) * 1000)
        if not text.strip():
            result.fail("Empty response from Anthropic", elapsed)
        else:
            result.ok(f"model={model} response='{text.strip()[:30]}'", elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result.fail(str(exc), elapsed)
    return result


async def check_qdrant() -> _Result:
    """Call get_collections() and verify the connection succeeds."""
    result = _Result("Qdrant", required=True)
    url = os.environ.get("QDRANT_URL", "")
    api_key = os.environ.get("QDRANT_API_KEY", "")

    if not url or "xxxx" in url:
        result.fail("QDRANT_URL not set or is placeholder", 0)
        return result

    t0 = time.perf_counter()
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=url, api_key=api_key, timeout=15)
        collections = await client.get_collections()
        await client.close()
        elapsed = int((time.perf_counter() - t0) * 1000)
        names = [c.name for c in collections.collections]
        collection_name = os.environ.get("QDRANT_COLLECTION", "documents")
        detail = f"{len(names)} collection(s): {names}"
        if collection_name not in names:
            detail += f" ⚠️ '{collection_name}' not found — run init_qdrant.py"
        result.ok(detail, elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result.fail(str(exc), elapsed)
    return result


async def check_redis() -> _Result:
    """PING, SET, GET, DELETE a test key to verify full round-trip."""
    result = _Result("Redis", required=True)
    redis_url = os.environ.get("REDIS_URL", "")

    if not redis_url or "xxxx" in redis_url:
        result.fail("REDIS_URL not set or is placeholder", 0)
        return result

    t0 = time.perf_counter()
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
        )
        # PING
        pong = await client.ping()
        assert pong is True, "PING did not return True"

        # SET / GET / DEL round-trip
        test_key = "autorag:preflight:test"
        test_val = "ok"
        await client.set(test_key, test_val, ex=30)
        retrieved = await client.get(test_key)
        assert retrieved == test_val, f"GET mismatch: expected '{test_val}', got '{retrieved}'"
        await client.delete(test_key)
        await client.aclose()

        elapsed = int((time.perf_counter() - t0) * 1000)
        result.ok("PING ✓ SET ✓ GET ✓ DEL ✓", elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result.fail(str(exc), elapsed)
    return result


async def check_postgres() -> _Result:
    """Run SELECT 1 via asyncpg to verify database connectivity."""
    result = _Result("PostgreSQL", required=True)
    db_url = os.environ.get("DATABASE_URL", "")

    if not db_url or "xxxx" in db_url:
        result.fail("DATABASE_URL not set or is placeholder", 0)
        return result

    # asyncpg requires 'postgresql://', not 'postgres://'
    db_url = db_url.replace("postgres://", "postgresql://", 1)

    t0 = time.perf_counter()
    try:
        import asyncpg
        conn = await asyncpg.connect(dsn=db_url, timeout=15, command_timeout=10)
        value = await conn.fetchval("SELECT 1")
        await conn.close()
        elapsed = int((time.perf_counter() - t0) * 1000)
        if value != 1:
            result.fail(f"SELECT 1 returned unexpected value: {value}", elapsed)
        else:
            # Parse host from DSN for display
            host = db_url.split("@")[-1].split("/")[0] if "@" in db_url else "unknown"
            result.ok(f"SELECT 1 = {value} | host={host}", elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result.fail(str(exc), elapsed)
    return result


async def check_s3() -> _Result:
    """list_buckets() to verify AWS credentials. Optional service."""
    result = _Result("AWS S3", required=False)
    bucket = os.environ.get("S3_BUCKET", "")
    key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

    if not all([bucket, key_id, secret]):
        result.skip("S3_BUCKET / AWS credentials not configured (optional)")
        return result

    t0 = time.perf_counter()
    try:
        import boto3
        from botocore.config import Config

        def _sync_check() -> list[str]:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=key_id,
                aws_secret_access_key=secret,
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                config=Config(connect_timeout=10, read_timeout=10),
            )
            resp = s3.list_buckets()
            return [b["Name"] for b in resp.get("Buckets", [])]

        buckets = await asyncio.to_thread(_sync_check)
        elapsed = int((time.perf_counter() - t0) * 1000)
        bucket_found = bucket in buckets
        detail = f"{len(buckets)} bucket(s) visible"
        if not bucket_found:
            detail += f" ⚠️ target bucket '{bucket}' not found"
        result.ok(detail, elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result.fail(str(exc), elapsed)
    return result


async def check_slack() -> _Result:
    """Call auth.test to verify the bot token. Optional service."""
    result = _Result("Slack", required=False)
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    signing = os.environ.get("SLACK_SIGNING_SECRET", "")

    if not token or not signing:
        result.skip("SLACK_BOT_TOKEN / SLACK_SIGNING_SECRET not configured (optional)")
        return result

    t0 = time.perf_counter()
    try:
        from slack_sdk.web.async_client import AsyncWebClient
        client = AsyncWebClient(token=token)
        auth = await client.auth_test()
        elapsed = int((time.perf_counter() - t0) * 1000)
        if not auth["ok"]:
            result.fail(f"auth_test returned ok=False: {auth.get('error', 'unknown')}", elapsed)
        else:
            bot_name = auth.get("user", "unknown")
            team = auth.get("team", "unknown")
            result.ok(f"bot={bot_name} team={team}", elapsed)
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result.fail(str(exc), elapsed)
    return result


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_all_checks() -> int:
    """
    Runs all checks sequentially (as specified) and prints a final summary.
    Returns 0 if all required services pass, 1 otherwise.
    """
    print()
    print(_bold("=" * 62))
    print(_bold("  AutoRAG — Pre-flight Service Check"))
    print(_bold("=" * 62))
    print()

    # Run checks sequentially in the specified order
    checks = [
        check_openai,
        check_anthropic,
        check_qdrant,
        check_redis,
        check_postgres,
        check_s3,
        check_slack,
    ]

    results: list[_Result] = []
    for check_fn in checks:
        r = await check_fn()
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(_bold("=" * 62))

    required_failures = [r for r in results if r.required and r.passed is False]
    optional_warnings = [r for r in results if not r.required and r.passed is False]
    skipped = [r for r in results if r.skipped]

    total_latency = sum(r.latency_ms for r in results if not r.skipped)

    if not required_failures:
        print(f"  🚀  {_bold(_green('ALL SYSTEMS GO'))} — ready to deploy")
    else:
        count = len(required_failures)
        failed_names = ", ".join(r.name for r in required_failures)
        print(f"  ⛔  {_bold(_red(f'{count} service(s) FAILED'))} — fix before deploying")
        print(f"      Failed: {failed_names}")

    if optional_warnings:
        warn_names = ", ".join(r.name for r in optional_warnings)
        print(f"  ⚠️   Optional services with errors: {warn_names}")

    if skipped:
        skip_names = ", ".join(r.name for r in skipped)
        print(f"  ℹ️   Skipped (not configured): {skip_names}")

    print(f"\n  Total check time: {total_latency}ms")
    print(_bold("=" * 62))
    print()

    return 1 if required_failures else 0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    exit_code = asyncio.run(run_all_checks())
    sys.exit(exit_code)
