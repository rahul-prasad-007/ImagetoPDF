# Image → Editable PDF (React UI + FastAPI + PaddleOCR)
# Single container: serves API at /api and SPA at /

FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html vite.config.js ./
COPY public ./public
COPY src ./src
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    OCR_LANG=auto \
    USE_PP_STRUCTURE=false \
    HYBRID_UNDERLAYS=true \
    CORS_ORIGINS=* \
    STATIC_DIR=/app/static

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY backend/app /app/app
COPY backend/fonts /app/fonts
COPY backend/.env.example /app/.env.example

COPY --from=frontend /frontend/dist /app/static

RUN mkdir -p /app/uploads /app/processed /app/results /app/debug /app/output \
    && printf '' > /app/uploads/.gitkeep \
    && printf '' > /app/processed/.gitkeep \
    && printf '' > /app/results/.gitkeep \
    && printf '' > /app/debug/.gitkeep \
    && printf '' > /app/output/.gitkeep

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
