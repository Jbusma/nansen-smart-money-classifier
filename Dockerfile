# Multi-stage build: Python API + Streamlit

# ── Stage 1: Python base ──────────────────────────────────────────────
FROM python:3.11-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir .

COPY src/ src/
COPY models/artifacts/ models/artifacts/

# ── Stage 2: API server ───────────────────────────────────────────────
FROM python-base AS api

EXPOSE 8000
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 3: Streamlit dashboard ──────────────────────────────────────
FROM python-base AS dashboard

EXPOSE 8501
CMD ["streamlit", "run", "src/serving/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
