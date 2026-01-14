<div align="center">

# 🧠 Synapse

### Instant Document Insights

[![CI](https://github.com/mugnihidayah/synapse-instant-document-insight/workflows/CI/badge.svg)](https://github.com/mugnihidayah/synapse-instant-document-insight/actions)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Production-Ready RAG (Retrieval-Augmented Generation) API for intelligent document Q&A**

[Features](#-features) • [Quick Start](#-quick-start) • [API Docs](#-api-documentation) • [Docker](#-docker) • [Tech Stack](#-tech-stack)

</div>

---

## ✨ Features

| Feature                     | Description                                |
| --------------------------- | ------------------------------------------ |
| 📄 **Multi-format Support** | PDF, DOCX, and TXT documents               |
| 🚀 **REST API**             | Production-ready FastAPI with Swagger docs |
| ⚡ **Streaming Responses**  | Real-time SSE streaming like ChatGPT       |
| 🌐 **Bilingual**            | Indonesian and English responses           |
| 💬 **Session Management**   | Multi-user isolated sessions               |
| 🐳 **Docker Ready**         | Containerized deployment                   |
| 🔄 **CI/CD**                | Automated testing with GitHub Actions      |
| ✅ **73% Test Coverage**    | Unit and integration tests                 |
| 🎛️ **Type Safe**            | Full type hints with MyPy                  |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/mugnihidayah/synapse-instant-document-insight.git
cd synapse-instant-document-insight

echo "GROQ_API_KEY=your_key" > .env
docker compose up -d

# Access: http://localhost:8000/docs
```

### Option 2: Local Development

```bash
git clone https://github.com/mugnihidayah/synapse-instant-document-insight.git
cd synapse-instant-document-insight

python -m venv venv && source venv/bin/activate
pip install -e ".[dev,api]"

cp .env.example .env  # Edit with your API keys

# Run API
uvicorn src.api.main:app --reload

# Or Streamlit UI
streamlit run app.py
```

---

## 📡 API Documentation

**Base URL:** `http://localhost:8000/api/v1`

| Method | Endpoint                  | Description       |
| ------ | ------------------------- | ----------------- |
| `POST` | `/documents/sessions`     | Create session    |
| `GET`  | `/documents/session/{id}` | Get session info  |
| `POST` | `/documents/upload/{id}`  | Upload documents  |
| `POST` | `/query/{id}`             | Query (sync)      |
| `POST` | `/query/stream/{id}`      | Query (streaming) |

### Example

```bash
# Create session
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/documents/sessions | jq -r '.session_id')

# Upload
curl -X POST "http://localhost:8000/api/v1/documents/upload/$SESSION" -F "files=@doc.pdf"

# Query
curl -X POST "http://localhost:8000/api/v1/query/stream/$SESSION" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "language": "en"}'
```

**Swagger UI:** `http://localhost:8000/docs`

---

## 🐳 Docker

```bash
docker compose up -d      # Start
docker compose logs -f    # Logs
docker compose down       # Stop
```

---

## 🛠️ Tech Stack

| Backend     | AI/ML       | DevOps         | Testing      |
| ----------- | ----------- | -------------- | ------------ |
| FastAPI     | LangChain   | Docker         | Pytest       |
| Python 3.12 | Groq LLM    | GitHub Actions | MyPy         |
| Pydantic    | HuggingFace | CI/CD          | Ruff         |
| Uvicorn     | ChromaDB    |                | 73% Coverage |

---

## 📁 Project Structure

```
synapse-instant-document-insight/
├── src/
│   ├── core/           # Config, exceptions
│   ├── rag/            # RAG chain, prompts
│   ├── ingestion/      # Loaders, chunkers
│   └── api/            # FastAPI endpoints
├── tests/              # Unit tests
├── .github/workflows/  # CI/CD
├── app.py              # Streamlit UI
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 🧪 Development

```bash
pytest tests/ -v --cov=src    # Tests
ruff check src/               # Lint
mypy src/                     # Type check
```

---

## 🔧 Configuration

```env
GROQ_API_KEY=gsk_your_key          # Required
HUGGINGFACE_TOKEN=hf_your_token    # Optional
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**Built with ❤️ using FastAPI, LangChain & Docker**

</div>