# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências de SO mínimas pra chromadb (sqlite vector ext) e psycopg.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential libpq5 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Cria volume pra persistir o ChromaDB no Railway (mount em /data/chroma)
ENV CHROMA_DIR=/data/chroma
RUN mkdir -p /data/chroma

EXPOSE 8000

# Railway injeta $PORT — fallback 8000 pra dev local.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
