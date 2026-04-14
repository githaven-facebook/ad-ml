# Multi-stage Dockerfile: GPU training stage + CPU serving stage

# -----------------------------------------------------------------------
# Base: CUDA runtime for GPU training
# -----------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS training-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -e ".[dev]"

# Copy remaining source
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# -----------------------------------------------------------------------
# Training image target
# -----------------------------------------------------------------------
FROM training-base AS training

COPY Makefile ./
ENTRYPOINT ["python"]
CMD ["scripts/train_user_persona.py", "--help"]

# -----------------------------------------------------------------------
# CPU-only serving base
# -----------------------------------------------------------------------
FROM python:3.10-slim AS serving-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e .

COPY src/ ./src/

# -----------------------------------------------------------------------
# Serving image target
# -----------------------------------------------------------------------
FROM serving-base AS serving

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; ch = grpc.insecure_channel('localhost:50051'); ch.close()" || exit 1

ENTRYPOINT ["python", "-m", "ad_ml.serving.grpc_server"]
CMD ["--port", "50051", "--workers", "4"]
