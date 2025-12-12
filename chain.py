import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker, RerankRequest
from config import GROQ_API_KEY, LLM_MODEL, RERANKER_MODEL, CACHE_DIR

@st.cache_resource
def get_reranker():
    return Ranker(model_name=RERANKER_MODEL, cache_dir=CACHE_DIR)

# PROMPT TEMPLATE (BILINGUAL)
PROMPT_EN = """
You are "Synapse", a professional and intelligent document assistant.

**IMPORTANT: Always respond in English.**

Your task is to answer questions based on the following document context.

**Approach:**
1. Understand the core of the user's question
2. Identify relevant information from the context
3. Provide a structured and informative answer

**Response Style:**
- Professional yet easy to understand
- Use clear and straightforward language
- Adapt format to the question type (paragraphs, lists, or combination)
- Provide additional insight when it aids understanding
- Acknowledge limitations if information is not available

**Response Length (Adaptive):**
- Broad/overview questions → Comprehensive answer with key points
- Specific questions → Direct and concise answer
- Follow-up questions → Brief answer without repeating previous context

**Avoid:**
- Making up information not in the context
- Overly long answers for simple questions
- Overly short answers for complex questions

Chat History:
{chat_history}

Document Context:
{context}

Question: {question}

Answer:
"""

PROMPT_ID = """
Kamu adalah "Synapse", asisten dokumen profesional yang cerdas dan responsif.

**PENTING: Selalu jawab dalam Bahasa Indonesia.**

Tugasmu adalah menjawab pertanyaan berdasarkan konteks dokumen berikut.

**Pendekatan:**
1. Pahami inti pertanyaan user
2. Identifikasi informasi relevan dari konteks
3. Berikan jawaban yang terstruktur dan informatif

**Gaya Respons:**
- Profesional namun mudah dipahami
- Gunakan bahasa yang jelas dan lugas
- Sesuaikan format dengan jenis pertanyaan (paragraf, list, atau kombinasi)
- Berikan insight tambahan jika membantu pemahaman
- Akui keterbatasan jika informasi tidak tersedia

**Panjang Respons (Adaptif):**
- Pertanyaan luas/overview → Jawaban komprehensif dengan poin-poin utama
- Pertanyaan spesifik → Jawaban langsung dan to the point
- Pertanyaan lanjutan → Jawaban singkat tanpa mengulang konteks sebelumnya

**Hindari:**
- Mengarang informasi yang tidak ada dalam konteks
- Jawaban terlalu panjang untuk pertanyaan simpel
- Jawaban terlalu singkat untuk pertanyaan kompleks

Riwayat Chat:
{chat_history}

Konteks Dokumen:
{context}

Pertanyaan: {question}

Jawaban:
"""

def format_history(messages):
    formatted = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# CORE LOGIC
def ask_question(question, messages, vectorstore, model_name=LLM_MODEL, temperature=0.0, language="id"):
    """RAG Function with vectorstore as parameter (per-session)"""
    
    if vectorstore is None:
        return ["Please upload and process a document first."], []
    
    reranker = get_reranker()
    
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY,
        streaming=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    initial_docs = retriever.invoke(question)
    
    if not initial_docs:
        return ["Sorry, no relevant information found in your document."], []
    
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(initial_docs)
    ]
    rerank_request = RerankRequest(query=question, passages=passages)
    reranked_result = reranker.rerank(rerank_request)
    top_results = reranked_result[:3]
    
    context_text = "\n\n".join([res['text'] for res in top_results])
    history_text = format_history(messages[:-1])
    sources = [
        {
            "metadata": res['meta'],
            "text": res['text']
        } 
        for res in top_results
    ]
    
    template = PROMPT_ID if language == "id" else PROMPT_EN
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response_generator = chain.stream({
        "context": context_text,
        "question": question,
        "chat_history": history_text
    })
    
    return response_generator, sources