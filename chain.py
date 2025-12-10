import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from flashrank import Ranker, RerankRequest
from config import *

@st.cache_resource
def get_vectorstore():
    # Setup Embedding
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Setup Vector DB
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    return db

@st.cache_resource
def get_reranker():
    return Ranker(model_name=RERANKER_MODEL, cache_dir=CACHE_DIR)

# PROMPT TEMPLATE (BILINGUAL)
PROMPT_EN = """
You are "Synapse", a highly intelligent corporate document assistant.
Answer user questions based ONLY on the following document context and conversation history.
**Guidelines for your response:**
1. Provide a comprehensive and detailed answer, not just a single sentence.
2. Use bullet points or numbered lists if it helps organize the information.
3. Quote or paraphrase relevant sections from the context to support your answer.
4. If the question is broad (e.g., "What is this document about?"), provide a summary that covers multiple key aspects.
5. If the answer is not in the context, clearly state: "I don't have enough information in the documents provided."
6. Do NOT make up facts.

Chat History:
{chat_history}

Document Context:
{context}

Question: 
{question}

Detailed Answer:
"""

PROMPT_ID = """
Anda adalah "Synapse", asisten dokumen korporat yang cerdas.
Jawab pertanyaan pengguna HANYA berdasarkan konteks dokumen dan riwayat percakapan berikut.
**Panduan untuk jawabanmu:**
1. Berikan jawaban yang lengkap dan detail, bukan hanya satu kalimat.
2. Gunakan poin-poin atau daftar bernomor jika membantu mengorganisasi informasi.
3. Kutip atau parafrasekan bagian relevan dari konteks untuk mendukung jawabanmu.
4. Jika pertanyaannya luas (misalnya "Dokumen ini tentang apa?"), berikan ringkasan yang mencakup beberapa aspek kunci.
5. Jika jawaban tidak ada di konteks, nyatakan dengan jelas: "Saya tidak memiliki informasi yang cukup dalam dokumen yang disediakan."
6. JANGAN mengarang fakta.

Riwayat Chat:
{chat_history}

Konteks Dokumen:
{context}

Pertanyaan: 
{question}

Jawaban Detail:
"""

def format_history(messages):
    """Converting Streamlit message format to plain text format for prompts"""
    formatted = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

# CORE LOGIC
def ask_question(question, messages, model_name=LLM_MODEL, temperature=0.0, language="id"): # <--- Tambah language
    """
    Primary RAG Function with Streaming, Memory, and Multilingual support.
    """
    vectorstore = get_vectorstore()
    reranker = get_reranker()
    
    # Dynamic LLM Setup (According to User's Choice)
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY,
        streaming=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    initial_docs = retriever.invoke(question)
    
    if not initial_docs:
        return ["Sorry, the document was not found or the database is empty."], []
    # Reranking (Passages construction)
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(initial_docs)
    ]
    rerank_request = RerankRequest(query=question, passages=passages)
    reranked_result = reranker.rerank(rerank_request)
    top_results = reranked_result[:3]

    # Context & History
    context_text = "\n\n".join([res['text'] for res in top_results])
    history_text = format_history(messages[:-1])
    sources = [res['meta'] for res in top_results]
    template = PROMPT_ID if language == "id" else PROMPT_EN
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response_generator = chain.stream({
        "context": context_text,
        "question": question,
        "chat_history": history_text
    })
    
    return response_generator, sources