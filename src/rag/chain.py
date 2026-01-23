"""
RAG chain for document Q&A

Combines retrieval, reranking, and LLM for answering questions
"""

from collections.abc import Iterator
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.core.config import settings
from src.core.exceptions import RAGError
from src.rag.prompts import get_prompt
from src.rag.reranker import rerank_documents


def format_chat_history(messages: list[dict]) -> str:
    """
    Format chat messages into a string for prompt

    Args:
      messages: List of chat messages

    Returns:
      Formatted chat history string
    """
    formatted = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted


def format_documents(docs: list) -> str:
    """
    Format documents into a string for context

    Args:
      docs: List of langchain documents

    Returns:
      Formatted context string
    """
    return "\n\n".join([d.page_content for d in docs])


def ask_question(
    question: str,
    messages: list[dict],
    vectorstore,
    model_name: str | None = None,
    temperature: float = 0.3,
    language: str = "id",
) -> tuple[Iterator[str], list[dict[str, Any]]]:
    """
    Answer question using RAG

    Args:
      question: User question
      messages: chat history
      vectorstore: ChromaDB vectorstore instance
      model_name: LLM model name
      temperature: LLM temperature
      language: Response language

    Returns:
      Tuple of response generator and source documents

    Raises:
      RAGError: If vectorstore is None or retrieval fails
    """
    if vectorstore is None:
        raise RAGError(
            "No vectorstore available",
            details={"hint": "Please upload and process a document first"},
        )

    if model_name is None:
        model_name = settings.llm_model

    # Initialize LLM
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=SecretStr(settings.groq_api_key),
        streaming=True,
    )

    # retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_top_k})
    initial_docs = retriever.invoke(question)

    if not initial_docs:
        raise RAGError("No relevant documents found", details={"question": question})

    # prepare for reranking
    passages = [
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(initial_docs)
    ]

    # rerank documents
    top_results = rerank_documents(question, passages)

    # format context and history
    context_text = "\n\n".join([res["text"] for res in top_results])
    history_text = format_chat_history(messages[:-1])  # exclude current message

    # prepare source for return
    sources = [{"metadata": res["meta"], "page_content": res["text"]} for res in top_results]

    # build chain
    template = get_prompt(language)
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # stream response
    response_generator = chain.stream(
        {"context": context_text, "question": question, "chat_history": history_text}
    )

    return response_generator, sources


def create_rag_chain(
    model_name: str | None = None,
    temperature: float = 0.3,
    language: str = "id",
):
    """Create a RAG chain for question answering"""

    llm = ChatGroq(
        api_key=SecretStr(settings.groq_api_key),
        model=model_name or settings.llm_model,
        temperature=temperature,
        streaming=True,
    )

    system_prompt = get_prompt(language)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain
