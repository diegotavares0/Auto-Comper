# Produtora — Production Dockerfile for Railway
# Multi-stage build to keep image lean

FROM python:3.11-slim AS base

# System deps for soundfile (libsndfile) and librosa (ffmpeg optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/ backend/
COPY frontend/ frontend/
COPY run.py wsgi.py ./

# Create data directory for presets
RUN mkdir -p data/presets

# Railway injects $PORT at runtime
ENV PORT=8085
EXPOSE ${PORT}

# Production server: gunicorn with threaded workers for SSE support
CMD gunicorn wsgi:app \
    --bind 0.0.0.0:${PORT} \
    --workers 2 \
    --threads 4 \
    --worker-class gthread \
    --timeout 300 \
    --keep-alive 65 \
    --log-level info
