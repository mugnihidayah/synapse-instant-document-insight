FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    # Memory optimization
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_OFFLINE=0 \
    HF_HOME=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/st_cache \
    # Reduce memory from NumPy/BLAS
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Copy dependency files
COPY pyproject.toml README.md ./
COPY src/ src/
# Install CPU-only PyTorch first (MUCH smaller than full torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Install remaining dependencies
RUN pip install --no-cache-dir -e ".[api]"
# Clean up
RUN rm -rf /root/.cache/pip
# Verify
RUN python -c "import uvicorn; print('uvicorn OK')"
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1
EXPOSE ${PORT}
CMD python -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT