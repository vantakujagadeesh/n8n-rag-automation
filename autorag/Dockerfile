# Full path: /Users/vantakujagadeesh/Desktop/rag n8n/autorag/Dockerfile

# ── Stage: Runtime image ──────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent .pyc files and enable real-time logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Playwright requires this to locate browsers inside the container
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# gcc      : required to compile some Python C extensions (e.g. asyncpg)
# curl     : used by Docker HEALTHCHECK
# chromium : base browser; Playwright will install its own managed build
#            on top of the system one (install-deps requires system chromium)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
        chromium \
        # Playwright Chromium runtime deps (fonts, libs)
        libnss3 \
        libatk-bridge2.0-0 \
        libdrm2 \
        libxkbcommon0 \
        libgbm1 \
        libasound2 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libpango-1.0-0 \
        libcairo2 \
        libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of
# source code changes. Only re-runs pip install when requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Playwright browser install ────────────────────────────────────────────────
# Install the managed Chromium build used by Playwright at runtime.
# install-deps installs any missing OS-level library deps.
RUN playwright install chromium && \
    playwright install-deps chromium

# ── Application source ────────────────────────────────────────────────────────
COPY app/ ./app/
COPY scripts/ ./scripts/

# ── Non-root user (security best practice) ───────────────────────────────────
RUN adduser --disabled-password --gecos "" autorag && \
    chown -R autorag:autorag /app /ms-playwright
USER autorag

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Waits 30s for startup, then checks every 30s. 3 consecutive failures = unhealthy.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# 2 workers: safe for most Railway instance sizes (1 vCPU / 512 MB RAM).
# Increase to 4 workers when scaling to 2+ vCPUs.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
