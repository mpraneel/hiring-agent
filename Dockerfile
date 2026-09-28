# Multi-stage build: Node compiles the UI, Python serves it alongside the API.
# One image runs the whole app from a single origin, which is why production
# needs no CORS configuration.

# --- Stage 1: build the frontend -------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Manifests first so the dependency layer is cached independently of source.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# --- Stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim

# Keep Python from writing .pyc files and buffering logs, so container logs
# appear in order.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY core ./core

# The built UI. api/main.py serves this when the directory exists.
COPY --from=frontend /build/dist ./frontend/dist

# Run as a non-root user. Nothing in the image needs write access: uploads go to
# the system temp directory and are deleted after each request.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
