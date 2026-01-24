<div align="center">

# 🧠 Synapse

### Instant Document Insights

[![CI](https://github.com/mugnihidayah/synapse-instant-document-insight/workflows/CI/badge.svg)](https://github.com/mugnihidayah/synapse-instant-document-insight/actions)
[![Deploy](https://img.shields.io/badge/Railway-Deployed-blueviolet?style=for-the-badge&logo=railway)](https://synapse-instant-document-insight-production.up.railway.app)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A RAG API for document Q&A — upload your PDFs, ask questions, get answers with citations.**

[Features](#-features) • [Quick Start](#-quick-start) • [API Docs](#-api-documentation) • [Docker](#-docker) • [Tech Stack](#-tech-stack)

🚀 **Live Demo:** [synapse-instant-document-insight-production.up.railway.app](https://synapse-instant-document-insight-production.up.railway.app/docs)

</div>

---

## Features

| Feature                  | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| **Multi-format Support** | PDF, DOCX, TXT, more formats coming                              |
| **REST API**             | FastAPI with auto-generated Swagger docs                         |
| **Streaming**            | SSE streaming responses, ChatGPT-style                           |
| **Bilingual**            | Responds in Indonesian or English based on your preference       |
| **Session-based**        | Each user gets isolated document storage                         |
| **API Key Auth**         | SHA256 hashed keys, stored in PostgreSQL                         |
| **Rate Limiting**        | 50 queries/min per key (configurable)                            |
| **Vector Search**        | PostgreSQL + pgvector for similarity search (384-dim embeddings) |
| **JSON Logging**         | Structured logs for production debugging                         |
| **Dockerized**           | One command to run everything                                    |
| **Tested**               | Unit tests with Pytest, CI/CD with GitHub Actions                |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/mugnihidayah/synapse-instant-document-insight.git
cd synapse-instant-document-insight

# Add your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# Start API + PostgreSQL
docker compose up -d

# Open http://localhost:8000/docs
```

### Local Development

**Linux/macOS:**

```bash
git clone https://github.com/mugnihidayah/synapse-instant-document-insight.git
cd synapse-instant-document-insight

python -m venv venv && source venv/bin/activate
pip install -e ".[dev,api]"

# Start PostgreSQL only
docker compose up db -d

# Create tables
psql $DATABASE_URL < scripts/init.sql

# Run the API
uvicorn src.api.main:app --reload
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/mugnihidayah/synapse-instant-document-insight.git
cd synapse-instant-document-insight

python -m venv venv
venv\Scripts\activate
pip install -e ".[dev,api]"

# Start PostgreSQL only
docker compose up db -d

# Create tables (use psql or run manually in pgAdmin)
# Copy contents of scripts/init.sql and execute in your PostgreSQL client

# Run the API
uvicorn src.api.main:app --reload
```

---

## API Documentation

**Base URL (Local):** `http://localhost:8000/api/v1`

**Base URL (Production):** `https://synapse-instant-document-insight-production.up.railway.app/api/v1`

### Authentication

All endpoints except `/keys/` need an API key:

```bash
# First, create a key
curl -X POST localhost:8000/api/v1/keys/ \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'

# Response: {"api_key": "sk-abc123...", "key_id": "...", ...}
# Save that key! It won't be shown again.

# Then use it in all requests
curl -H "X-API-Key: sk-abc123..." localhost:8000/api/v1/documents/sessions
```

### Endpoints

| Method   | Endpoint                   | What it does                | Needs auth? |
| -------- | -------------------------- | --------------------------- | ----------- |
| `POST`   | `/keys/`                   | Get an API key              | No          |
| `GET`    | `/keys/`                   | List your keys (hashed)     | No          |
| `DELETE` | `/keys/{id}`               | Revoke a key                | No          |
| `POST`   | `/documents/sessions`      | Start a new session         | Yes         |
| `GET`    | `/documents/sessions/{id}` | Check session status        | Yes         |
| `DELETE` | `/documents/sessions/{id}` | Delete session and docs     | Yes         |
| `POST`   | `/documents/upload/{id}`   | Upload files to session     | Yes         |
| `POST`   | `/query/{id}`              | Ask a question              | Yes         |
| `POST`   | `/query/stream/{id}`       | Ask with streaming response | Yes         |

### Typical Flow

```bash
# 1. Get API key (do this once)
API_KEY=$(curl -s -X POST localhost:8000/api/v1/keys/ \
  -H "Content-Type: application/json" \
  -d '{"name": "test"}' | jq -r '.api_key')

# 2. Create a session
SESSION=$(curl -s -X POST localhost:8000/api/v1/documents/sessions \
  -H "X-API-Key: $API_KEY" | jq -r '.session_id')

# 3. Upload your document
curl -X POST "localhost:8000/api/v1/documents/upload/$SESSION" \
  -H "X-API-Key: $API_KEY" \
  -F "files=@quarterly-report.pdf"

# 4. Ask questions
curl -X POST "localhost:8000/api/v1/query/$SESSION" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue growth?", "language": "en"}'
```

**Swagger UI (Local):** http://localhost:8000/docs

**Swagger UI (Production):** https://synapse-instant-document-insight-production.up.railway.app/docs

---

## Rate Limits

Limits are per API key:

| Endpoint type | Limit     |
| ------------- | --------- |
| Queries       | 50/minute |
| Uploads       | 5/minute  |
| Session ops   | 20/minute |

Hit the limit? You'll get `429 Too Many Requests`. Wait a minute and retry.

---

## Docker

```bash
docker compose up -d        # Start everything
docker compose logs -f api  # Watch API logs
docker compose down         # Stop
```

---

## Tech Stack

**Backend:** FastAPI, SQLAlchemy, Pydantic, Uvicorn

**Database:** PostgreSQL with pgvector extension

**AI/ML:** LangChain, Groq LLM, HuggingFace embeddings (384-dim)

**Auth & Security:** SHA256 key hashing, slowapi rate limiting

**Logging:** structlog (JSON format)

**DevOps:** Docker, GitHub Actions CI

---

## Project Structure

```
synapse-instant-document-insight/
├── src/
│   ├── api/              # FastAPI app
│   │   ├── routes/       # Endpoint handlers (documents, query, keys)
│   │   ├── auth.py       # API key generation & validation
│   │   ├── dependencies.py
│   │   ├── rate_limiter.py
│   │   ├── schemas.py    # Pydantic models
│   │   └── main.py       # App factory
│   ├── core/             # Shared utilities
│   │   ├── config.py     # Settings (env vars)
│   │   ├── exceptions.py
│   │   └── logger.py     # Structured logging
│   ├── db/               # Database layer
│   │   ├── connection.py # Async engine & session
│   │   └── models.py     # SQLAlchemy ORM models
│   ├── ingestion/        # Document processing
│   │   ├── loaders.py    # PDF, DOCX, TXT parsers
│   │   ├── chunkers.py   # Text splitting
│   │   └── pgvector_store.py  # Vector storage
│   └── rag/              # AI/ML logic
│       ├── chain.py      # LangChain RAG chain
│       └── prompts.py    # System prompts (ID/EN)
├── tests/                # Pytest test suite
├── scripts/
│   └── init.sql          # Database schema
├── .github/workflows/    # CI/CD
├── app.py                # Streamlit UI (optional)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Known Limitations

- **File size:** Large files load into memory, no streaming upload yet
- **Session expiry:** Sessions auto delete after 24 hours
- **No chat history:** Each query is independent, no multi-turn conversations
- **Single similarity metric:** Cosine distance only (no hybrid search)
- **Embedding model:** Fixed at 384 dimensions (multilingual-MiniLM-L12-v2)

---

## Development

```bash
pytest tests/ -v --cov=src    # Run tests
ruff check src/               # Lint
mypy src/                     # Type check
```

---

## Environment Variables

```env
# Required
GROQ_API_KEY=gsk_your_key
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db

# Optional
HUGGINGFACE_TOKEN=hf_xxx      # For private models
LOG_LEVEL=info                # debug, info, warning, error
PORT=8000
```

---

## License

MIT — do whatever you want.

---

<div align="center">

**Built with FastAPI, LangChain, PostgreSQL & Docker**

</div>
