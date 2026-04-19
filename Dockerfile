# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install fastapi uvicorn[standard]

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY churn_model/ ./churn_model/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Health check — calls the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Run with 2 Uvicorn workers (tune based on available CPU cores)
CMD ["uvicorn", "churn_model.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
