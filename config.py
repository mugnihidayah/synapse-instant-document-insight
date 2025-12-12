import os
from dotenv import load_dotenv

load_dotenv()

# Directory
CACHE_DIR = "./opt"

# API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# MODEL CONFIGURATION
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
AVAILABLE_LLMS = [
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct-0905",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b"
]