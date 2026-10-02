# =============================================================
# Dockerfile — PISA (Zomato Hyperpure B2B)
# One image used for both the Streamlit dashboard and the FastAPI service
# (docker-compose.yml runs it twice with different commands).
# =============================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8501 \
    PISA_ENV=production

WORKDIR /app

# curl: healthcheck. build-essential: fallback for any dependency without a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore keeps secrets.toml, .env and local data/models out of the image
COPY . .

# Bake in data, trained models and the DuckDB warehouse at build time (MLflow is not
# installed here, so tracking is skipped). docker-compose mounts named volumes over
# data/ and models/; Docker seeds an empty named volume from these files on first run.
RUN python pipeline.py

RUN useradd --create-home pisa && chown -R pisa /app
USER pisa

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
