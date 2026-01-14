FROM python:3.12-slim
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Copy dependency files
COPY pyproject.toml README.md ./
COPY src/ src/
# Install Python dependencies
RUN pip install --no-cache-dir -e ".[api]"
# Verify uvicorn installed
RUN python -c "import uvicorn; print('uvicorn OK')"
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1
# Expose port
EXPOSE ${PORT}
# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]