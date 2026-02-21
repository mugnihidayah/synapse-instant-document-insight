<div align="center">

# Synapse

### Instant Document Insights

[![CI](https://github.com/mugnihidayah/synapse-instant-document-insight/workflows/CI/badge.svg)](https://github.com/mugnihidayah/synapse-instant-document-insight/actions)
[![Deploy](https://img.shields.io/badge/HF%20Spaces-Deployed-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://mugnihidayah-synapse-rag-api.hf.space)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A multimodal RAG API for document Q&A: upload docs, ask questions, get grounded answers with citations.**

[Features](#features) • [Quick Start](#quick-start) • [API](#api-documentation) • [Eval](#evaluation) • [Tech Stack](#tech-stack)

API Docs: [mugnihidayah-synapse-rag-api.hf.space/docs](https://mugnihidayah-synapse-rag-api.hf.space/docs)

</div>

---

## Features

- Multimodal ingestion: `PDF`, `DOCX`, `TXT`, image files (`PNG/JPG/JPEG/WEBP`) with OCR.
- Async ingestion pipeline: upload queue with status (`queued`, `processing`, `ready`, `failed`).
- Retrieval upgrades: hybrid search (vector + keyword), reranking, dynamic `top_k`, MMR diversification.
- Query quality: contextualization, query rewrite, strict grounding guardrail, richer citations.
- Metadata filters at query time: by source, page range, chunk type, content origin.
- Session tools: session status, paginated chunk listing, session export (`markdown`/`json`).
- Product signals: feedback endpoint and usage analytics with daily query quota.
- Secure API key auth, rate limiting, structured JSON logging.

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/mugnihidayah/synapse-instant-document-insight.git
cd synapse-instant-document-insight

# set required env vars
echo "GROQ_API_KEY=gsk_your_key_here" > .env

docker compose up -d
# open http://localhost:8000/docs
```

### Local Development

```bash
git clone https://github.com/mugnihidayah/synapse-instant-document-insight.git
cd synapse-instant-document-insight

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev,api]"
docker compose up db -d

# initialize/update schema (run this on your DB, including Neon)
psql "$DATABASE_URL" -f scripts/init.sql

uvicorn src.api.main:app --reload
```

---

## API Documentation

Base URL (local): `http://localhost:8000/api/v1`

### Authentication

- `POST /keys/` does not require auth.
- All other endpoints require header: `X-API-Key: sk-...`

Create API key:

```bash
curl -X POST localhost:8000/api/v1/keys/ \
  -H "Content-Type: application/json" \
  -d '{"name":"my-app"}'
```

### Endpoints

| Method | Endpoint | Description | Auth |
| --- | --- | --- | --- |
| `POST` | `/keys/` | Create API key | No |
| `GET` | `/keys/` | Get current key metadata | Yes |
| `DELETE` | `/keys/{key_id}` | Revoke current key | Yes |
| `POST` | `/documents/sessions` | Create session | Yes |
| `GET` | `/documents/sessions/{session_id}` | Get session info + ingestion status | Yes |
| `GET` | `/documents/sessions/{session_id}/documents` | Paginated chunk list | Yes |
| `DELETE` | `/documents/sessions/{session_id}` | Delete session | Yes |
| `POST` | `/documents/upload/{session_id}` | Upload documents (async by default) | Yes |
| `GET` | `/documents/supported-formats` | Supported upload formats | No |
| `POST` | `/query/{session_id}` | Non-streaming query | Yes |
| `POST` | `/query/stream/{session_id}` | Streaming query (SSE) | Yes |
| `POST` | `/insights/feedback/{session_id}` | Submit answer feedback | Yes |
| `GET` | `/insights/usage` | Usage + quota summary | Yes |
| `GET` | `/insights/export/{session_id}` | Export chat history | Yes |

### Typical Flow (Async Upload)

```bash
# 1) Create key
API_KEY=$(curl -s -X POST localhost:8000/api/v1/keys/ \
  -H "Content-Type: application/json" \
  -d '{"name":"demo"}' | jq -r '.api_key')

# 2) Create session
SESSION=$(curl -s -X POST localhost:8000/api/v1/documents/sessions \
  -H "X-API-Key: $API_KEY" | jq -r '.session_id')

# 3) Upload (async_mode=true default)
curl -X POST "localhost:8000/api/v1/documents/upload/$SESSION" \
  -H "X-API-Key: $API_KEY" \
  -F "files=@report.pdf"

# 4) Poll session status until ingestion_status=ready
curl -H "X-API-Key: $API_KEY" \
  "localhost:8000/api/v1/documents/sessions/$SESSION"

# 5) Query
curl -X POST "localhost:8000/api/v1/query/$SESSION" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the revenue growth?","language":"en"}'
```

### Query Controls (Optional)

Example payload with filters/debug:

```json
{
  "question": "Summarize key risks",
  "language": "en",
  "top_k": 8,
  "rerank_top_k": 3,
  "include_debug": true,
  "strict_grounding": true,
  "enable_query_rewrite": true,
  "filters": {
    "sources": ["risk-report.pdf"],
    "page_from": 2,
    "page_to": 12,
    "chunk_types": ["content"],
    "content_origin": "text+ocr"
  }
}
```

### Upload Controls (Optional Query Params)

- `async_mode` (default `true`)
- `enable_ocr` (default from config)
- `extract_tables` (default from config)

Example:

```bash
curl -X POST "localhost:8000/api/v1/documents/upload/$SESSION?async_mode=false&enable_ocr=true&extract_tables=true" \
  -H "X-API-Key: $API_KEY" \
  -F "files=@scanned.pdf"
```

---

## Rate Limits

Per API key:

- Query: `50/minute`
- Upload: `5/minute`
- Session operations: `20/minute`

Daily soft quota:

- Query quota also tracked per day (`USAGE_DAILY_QUERY_QUOTA`, default `1000`).

---

## Evaluation

Lightweight eval harness:

```bash
python scripts/eval/eval_harness.py --input scripts/eval/sample_predictions.jsonl
```

Metrics:

- `exact_match`
- `token_f1`
- `grounding_score`
- `source_recall`

---

## Tech Stack

- Backend: FastAPI, SQLAlchemy, Pydantic, Uvicorn
- Database: PostgreSQL + pgvector
- AI/ML: LangChain, Groq LLM, HuggingFace embeddings, Cohere/local reranker
- OCR: `rapidocr_onnxruntime` (RapidOCR)
- Search: Hybrid retrieval + reranking + MMR
- DevOps: Docker, GitHub Actions, Hugging Face Spaces

---

## Known Limitations

- Async ingestion uses in-process background tasks (no distributed worker yet).
- Upload still buffers files in memory before processing.
- OCR quality depends on scan quality and OCR runtime availability.
- Sessions expire automatically after 24 hours.
- Free-tier hosting can have cold starts.

---

## Development

```bash
pytest tests/ -v --cov=src
ruff check src/
mypy src/
```

---

## Environment Variables

```env
# Required
GROQ_API_KEY=gsk_your_key
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/synapse_db

# Optional
COHERE_API_KEY=
HUGGINGFACE_TOKEN=
RERANKER_PROVIDER=cohere
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_MODEL=ms-marco-MiniLM-L-12-v2
LOG_LEVEL=info
PORT=8000
DEBUG=false
CACHE_DIR=./opt
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval / quality
RETRIEVAL_TOP_K=10
RETRIEVAL_FETCH_K=20
DYNAMIC_TOP_K_MIN=4
DYNAMIC_TOP_K_MAX=10
RERANK_TOP_K=3
USE_HYBRID_SEARCH=true
HYBRID_VECTOR_WEIGHT=0.5
HYBRID_KEYWORD_WEIGHT=0.5
USE_MMR=true
MMR_LAMBDA=0.7
GROUNDEDNESS_THRESHOLD=0.15
QUERY_REWRITE_ENABLED=true

# Ingestion
INGESTION_ASYNC_DEFAULT=true
ENABLE_OCR=true
ENABLE_TABLE_EXTRACTION=true
MAX_UPLOAD_FILE_SIZE_MB=25

# Analytics / export
USAGE_DAILY_QUERY_QUOTA=1000
EXPORT_MAX_MESSAGES=200
```

---

## License

MIT

