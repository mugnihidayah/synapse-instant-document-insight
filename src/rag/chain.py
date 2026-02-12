"""
RAG chain for document Q&A

Creates LangChain pipeline for question answering with streaming support
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.core.config import settings
from src.rag.prompts import get_prompt


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
