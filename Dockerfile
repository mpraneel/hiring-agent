# Runtime image for the hiring agent API.
FROM python:3.12-slim

# Keep Python from writing .pyc files and buffering logs, so container logs
# appear in order.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies are copied and installed before the source so that editing code
# does not invalidate the cached dependency layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY core ./core

# Run as a non-root user. Nothing in the image needs write access: uploads go to
# the system temp directory and are deleted after each request.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# No shell form, so uvicorn is PID 1 and receives SIGTERM directly.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
