<div align="center">

# 🧠 Synapse

### Instant Document Insights

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C?style=for-the-badge)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An Advanced RAG (Retrieval-Augmented Generation) System for intelligent document Q&A**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Contributing](#-contributing)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-format Support** | Upload and process PDF, DOCX, and TXT documents |
| 🌐 **Bilingual Response** | Supports both Indonesian and English responses |
| ⚡ **Streaming Output** | Real-time response generation for better UX |
| 💬 **Chat Memory** | Context-aware conversations with chat history |
| 🎛️ **Model Selection** | Choose from multiple LLM models |
| 🔧 **Temperature Control** | Adjust creativity level of responses |
| 🔄 **Database Management** | Easy reset and document management |

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- Groq API Key ([Get it here](https://console.groq.com))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/synapse.git
   cd synapse
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

1. **Upload Documents** — Use the sidebar to upload PDF, DOCX, or TXT files
2. **Process Documents** — Click "Document Process (Ingest)" to index your documents
3. **Ask Questions** — Type your question in the chat input
4. **View Sources** — Expand "Reference Sources" to see where the answer came from

### Configuration Options

| Option | Description |
|--------|-------------|
| **AI Model** | Select from available LLM models |
| **Temperature** | 0.0 (focused) to 1.0 (creative) |
| **Language** | Choose response language (ID/EN) |

## 🛠️ Tech Stack

<table>
<tr>
<td align="center"><b>Frontend</b></td>
<td align="center"><b>Backend</b></td>
<td align="center"><b>AI/ML</b></td>
<td align="center"><b>Database</b></td>
</tr>
<tr>
<td>

- Streamlit

</td>
<td>

- Python
- LangChain

</td>
<td>

- Groq LLM
- HuggingFace Embeddings
- FlashRank Reranker

</td>
<td>

- ChromaDB

</td>
</tr>
</table>

## 📁 Project Structure

```
synapse/
├── app.py              # Main Streamlit application
├── chain.py            # RAG chain logic & LLM integration
├── ingest.py           # Document processing & ingestion
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not tracked)
├── data/               # Uploaded documents (not tracked)
└── vectorstore/        # ChromaDB storage (not tracked)
```

## 🔧 Configuration

All configuration options are centralized in `config.py`:

```python
# Paths
DATA_PATH = "./data"
DB_PATH = "./vectorstore"

# Models
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using Streamlit & LangChain**

</div>
