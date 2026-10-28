# syntax=docker/dockerfile:1
# Multi-stage Linux-compatible production Dockerfile (AMD64 & ARM64 compatible)

FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip build wheel && \
    pip wheel --no-deps --wheel-dir /build/wheels . && \
    pip wheel --wheel-dir /build/wheels \
        "fastapi>=0.115.0,<1.0.0" \
        "uvicorn[standard]>=0.32.0,<1.0.0" \
        "python-multipart>=0.0.12,<1.0.0" \
        "pydantic>=2.9.0,<3.0.0" \
        "pydantic-settings>=2.6.0,<3.0.0" \
        "sqlalchemy>=2.0.36,<3.0.0" \
        "pikepdf>=9.4.0,<10.0.0" \
        "pypdfium2>=4.30.0,<5.0.0" \
        "pillow>=11.0.0,<12.0.0" \
        "opencv-python-headless>=4.10.0.84,<5.0.0" \
        "numpy>=2.0.0,<3.0.0" \
        "fonttools>=4.55.0,<5.0.0" \
        "psutil>=6.1.0,<7.0.0" \
        "httpx>=0.27.2,<1.0.0" \
        "onnxruntime>=1.20.0,<2.0.0" \
        "tokenizers>=0.20.0,<1.0.0"

# --- Production Runtime Stage ---
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TEMP_STORAGE_PATH=/data/ocr \
    UNICODE_FONT_PATH=/app/resources/fonts/ArchiveUnicode-Regular.ttf \
    SRGB_ICC_PATH=/app/resources/icc/sRGB.icc

# Install native CPU OCR and PDF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-tur \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    libqpdf29 \
    poppler-utils \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged service user (FR-17)
RUN groupadd --gid 10001 ocrgroup && \
    useradd --uid 10001 --gid ocrgroup --shell /usr/sbin/nologin --create-home ocruser && \
    mkdir -p /app /data/ocr/jobs && \
    chown -R ocruser:ocrgroup /app /data/ocr

WORKDIR /app

COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

COPY --chown=ocruser:ocrgroup resources /app/resources

USER ocruser:ocrgroup

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health/live || exit 1

CMD ["ocr-api"]
