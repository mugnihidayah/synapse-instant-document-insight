"""
Query contextualization for multi-turn conversations.

Transform ambiguous queries like "Explain in more detail" into comprehensive queries based on chat history.
"""

from langchain_groq import ChatGroq

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


CONTEXTUALIZE_PROMPT = """Given the conversation history and 
the user's latest question,
create a standalone question that can be 
understood without needing to see the conversation history.

If the question is already clear, return it as is.
Do not answer the question, only reformulate if necessary.

Conversation History:
{chat_history}

Latest Question: {question}

Reformulated Question:"""


async def contextualize_query(
    question: str,
    chat_history: str,
    model_name: str | None = None,
) -> str:
    """
    Contextualize queries based on chat history.

    Args:
        question: User's question
        chat_history: Formatted chat history
        model_name: Model LLM

    Returns:
        Contextualized question
    """
    # If there is no history, return the question directly.
    if not chat_history:
        return question

    try:
        llm = ChatGroq(model=model_name or settings.llm_model, temperature=0)

        prompt = CONTEXTUALIZE_PROMPT.format(chat_history=chat_history, question=question)

        response = await llm.ainvoke(prompt)
        contextualized = response.content.strip()

        logger.info(
            "query_contextualized", original=question[:50], contextualized=contextualized[:50]
        )

        return contextualized

    except Exception as e:
        logger.error("contextualize_failed", error=str(e))
        # fallback to original question
        return question
