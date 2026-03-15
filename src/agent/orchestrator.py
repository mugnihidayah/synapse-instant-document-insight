"""
Agent orchestrator for Agentic RAG.

Implements a ReAct-style reasoning loop where the LLM agent
autonomously decides which tools to use and when to stop.
"""

import asyncio
import json
import re
import uuid

from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.state import AgentResult, AgentStep
from src.agent.tools import (
    TOOL_DESCRIPTIONS,
    format_retrieved_docs,
    tool_analyze_sources,
    tool_compare_sources,
    tool_refine_query,
    tool_retrieve,
    tool_summarize_context,
)
from src.core.config import settings
from src.core.logger import get_logger
from src.rag.grounding import is_grounded
from src.rag.prompts import get_agent_prompt
from src.rag.retrieval_utils import build_snippet

logger = get_logger(__name__)

# Delay between LLM calls to avoid Groq RPM limits
THROTTLE_DELAY_SECONDS = 0.5


def _build_tool_descriptions_text() -> str:
    """Build a human-readable tool description string for the system prompt."""
    parts = []
    for tool in TOOL_DESCRIPTIONS:
        params = tool.get("parameters", {})
        param_text = ""
        if params:
            param_items = [f"  - {k}: {v}" for k, v in params.items()]
            param_text = "\nParameters:\n" + "\n".join(param_items)
        parts.append(f"**{tool['name']}**: {tool['description']}{param_text}")
    return "\n\n".join(parts)


AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Search the uploaded documents with a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sources",
            "description": "Evaluate if retrieved sources are sufficient.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_context",
            "description": "Summarize the retrieved context.",
            "parameters": {"type": "object", "properties": {"focus": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refine_query",
            "description": "Generate a better search query.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sources",
            "description": "Cross-reference information across retrieved sources.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _parse_agent_response(text: str) -> dict | None:
    """
    Parse agent response to extract tool call or final answer.

    Handles both clean JSON and JSON embedded in markdown code blocks.
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    return None


def _build_sources_from_docs(docs: list[LangchainDocument], query: str) -> list[dict]:
    """Convert retrieved LangchainDocuments to SourceItem list."""
    sources = []
    for doc in docs:
        meta = dict(doc.metadata)
        chunk_id = str(meta.pop("id", ""))
        document_id = meta.get("document_id")

        sources.append(
            {
                "text": doc.page_content,
                "snippet": build_snippet(doc.page_content, query),
                "score": float(meta.get("rerank_score", 0.5)),
                "chunk_id": chunk_id,
                "document_id": str(document_id) if document_id else None,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "metadata": meta,
            }
        )
    return sources


async def run_agent(
    question: str,
    session_id: uuid.UUID,
    db: AsyncSession,
    language: str = "id",
    model_name: str | None = None,
    max_iterations: int | None = None,
    temperature: float | None = None,
    filters: dict | None = None,
    chat_history_str: str = "",
) -> AgentResult:
    """
    Run the agentic RAG loop.

    ReAct cycle:
    1. Agent receives question + available tools
    2. Agent thinks about what to do (Thought)
    3. Agent picks a tool + arguments (Action)
    4. System executes tool, returns result (Observation)
    5. Repeat until agent produces Final Answer or hits max iterations
    """
    max_iter: int = (
        int(max_iterations)
        if max_iterations is not None
        else int(getattr(settings, "agent_max_iterations", 5))
    )
    agent_temp = (
        temperature if temperature is not None else getattr(settings, "agent_temperature", 0.1)
    )
    model = model_name or settings.llm_model

    llm = ChatGroq(
        api_key=SecretStr(settings.groq_api_key),
        model=model,
        temperature=agent_temp,
    )

    # Bind native tools to prevent API crashes on aggressive models
    try:
        llm_with_tools = llm.bind_tools(AGENT_TOOLS_SCHEMA)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("failed_to_bind_tools", error=str(e))
        llm_with_tools = llm

    # Build system prompt with tool descriptions
    tool_desc_text = _build_tool_descriptions_text()
    system_prompt = get_agent_prompt(language).format(
        tool_descriptions=tool_desc_text,
        max_iterations=max_iter,
    )

    # Conversation messages for the agent
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
    ]

    # Add chat history context if available
    if chat_history_str:
        messages.append(SystemMessage(content=f"Previous conversation:\n{chat_history_str}"))

    messages.append(HumanMessage(content=question))

    # Agent state
    steps: list[AgentStep] = []
    all_retrieved_docs: list[LangchainDocument] = []
    current_context = ""
    iterations = 0

    for iteration in range(max_iter):
        iterations = iteration + 1

        # Throttle to avoid Groq RPM limits
        if iteration > 0:
            await asyncio.sleep(THROTTLE_DELAY_SECONDS)

        # Get agent response
        try:
            response = await llm_with_tools.ainvoke(messages)
            content = response.content
            agent_text = content.strip() if isinstance(content, str) else str(content)
        except Exception as e:
            logger.error("agent_llm_failed", error=str(e), iteration=iteration)
            steps.append(
                AgentStep(
                    step_type="thought",
                    content=f"Error calling LLM: {e}. Falling back to direct answer.",
                )
            )
            response = AIMessage(content="")  # Fallback to avoid unbounded variable
            agent_text = ""
            break

        parsed = None
        tool_call_id = None

        # Determine if model used native tool calling or text JSON
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            tool_call = tool_calls[0]
            parsed = {"tool": tool_call["name"], "arguments": tool_call["args"]}
            tool_call_id = tool_call.get("id")
            messages.append(response)  # Append AIMessage with tool_calls
        else:
            parsed = _parse_agent_response(agent_text)
            messages.append(AIMessage(content=agent_text))

        if parsed is None:
            # Agent responded with plain text — treat as thought + final answer
            steps.append(AgentStep(step_type="thought", content=agent_text))

            # If it looks like a final answer (long enough, no tool keywords)
            if len(agent_text) > 100:
                steps.append(
                    AgentStep(
                        step_type="final_answer",
                        content=agent_text,
                    )
                )
                break
            else:
                # Ask agent to format properly
                messages.append(
                    HumanMessage(
                        content="Please respond with a JSON tool call or a final_answer JSON."
                    )
                )
                continue

        # Check if it's a final answer
        if "final_answer" in parsed:
            answer = parsed["final_answer"]
            steps.append(
                AgentStep(
                    step_type="final_answer",
                    content=answer,
                )
            )
            break

        # It's a tool call
        tool_name = parsed.get("tool", "")
        tool_args = parsed.get("arguments", {})

        steps.append(
            AgentStep(
                step_type="action",
                content=f"Using tool: {tool_name}",
                tool_name=tool_name,
                tool_input=tool_args,
            )
        )

        # Execute the tool
        tool_result = await _execute_tool(
            tool_name=tool_name,
            tool_args=tool_args,
            db=db,
            session_id=session_id,
            question=question,
            current_context=current_context,
            retrieved_docs=all_retrieved_docs,
            filters=filters,
        )

        # Update state based on tool results
        if tool_name == "retrieve" and isinstance(tool_result, dict):
            new_docs = tool_result.get("_docs", [])
            all_retrieved_docs.extend(new_docs)
            current_context = format_retrieved_docs(all_retrieved_docs)
            display_result = tool_result.get("display", "No results")
        else:
            display_result = str(tool_result)

        steps.append(
            AgentStep(
                step_type="observation",
                content=display_result[:2000],  # Truncate long observations
                tool_name=tool_name,
                tool_output=display_result[:2000],
            )
        )

        # Add to conversation for next iteration
        if tool_call_id:
            messages.append(
                ToolMessage(
                    content=f"Tool result:\n{display_result[:2000]}", tool_call_id=tool_call_id
                )
            )
        else:
            messages.append(HumanMessage(content=f"Tool result:\n{display_result[:2000]}"))

    # Build final result
    final_answer = ""
    for step in reversed(steps):
        if step.step_type == "final_answer":
            final_answer = step.content
            break

    # If no final answer was produced, generate one from context
    if not final_answer and current_context:
        final_answer = await _fallback_generate(
            llm, question, current_context, language, chat_history_str
        )
        steps.append(
            AgentStep(
                step_type="final_answer",
                content=final_answer,
            )
        )

    if not final_answer:
        final_answer = (
            "I could not find sufficient information in the documents to answer this question."
            if language == "en"
            else "Saya tidak menemukan informasi yang cukup dalam dokumen untuk menjawab pertanyaan ini."
        )
        steps.append(
            AgentStep(
                step_type="final_answer",
                content=final_answer,
            )
        )

    # Grounding check
    source_texts = [doc.page_content for doc in all_retrieved_docs]
    grounded, grounding_score = is_grounded(final_answer, source_texts)

    # Build sources
    sources = _build_sources_from_docs(all_retrieved_docs, question)

    logger.info(
        "agent_completed",
        iterations=iterations,
        steps=len(steps),
        sources=len(sources),
        grounded=grounded,
        grounding_score=grounding_score,
    )

    return AgentResult(
        answer=final_answer,
        steps=steps,
        sources=sources,
        iterations=iterations,
        model_used=model,
        grounded=grounded,
        grounding_score=grounding_score,
        rewritten_query=question,
    )


async def _execute_tool(
    tool_name: str,
    tool_args: dict,
    db: AsyncSession,
    session_id: uuid.UUID,
    question: str,
    current_context: str,
    retrieved_docs: list[LangchainDocument],
    filters: dict | None = None,
) -> dict | str:
    """Dispatch and execute a tool by name."""
    try:
        if tool_name == "retrieve":
            query = tool_args.get("query", question)
            top_k = int(tool_args.get("top_k", 5))
            docs = await tool_retrieve(db, session_id, query, top_k=top_k, filters=filters)
            display = format_retrieved_docs(docs)
            return {"display": display, "_docs": docs}

        elif tool_name == "analyze_sources":
            analyze_result = await tool_analyze_sources(question, retrieved_docs)
            return json.dumps(analyze_result, ensure_ascii=False)

        elif tool_name == "summarize_context":
            focus = tool_args.get("focus", "")
            summary_result = await tool_summarize_context(current_context, focus=focus)
            return summary_result

        elif tool_name == "refine_query":
            reason = tool_args.get("reason", "")
            refined_result = await tool_refine_query(question, current_context, reason=reason)
            return f"Refined query: {refined_result}"

        elif tool_name == "compare_sources":
            compare_result = await tool_compare_sources(question, retrieved_docs)
            return compare_result

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as e:
        logger.error("tool_execution_failed", tool=tool_name, error=str(e))
        return f"Tool error: {e}"


async def _fallback_generate(
    llm: ChatGroq,
    question: str,
    context: str,
    language: str,
    chat_history: str = "",
) -> str:
    """Fallback: generate answer directly from context when agent loop ends without final answer."""
    lang_instruction = (
        "Respond in English." if language == "en" else "Jawab dalam Bahasa Indonesia."
    )

    prompt = (
        f"Based on the following document context, answer the question.\n"
        f"{lang_instruction}\n\n"
        f"Context:\n{context[:4000]}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    try:
        response = await llm.ainvoke(prompt)
        content = response.content
        return content.strip() if isinstance(content, str) else str(content)
    except Exception as e:
        logger.error("fallback_generate_failed", error=str(e))
        if language == "en":
            return "Failed to generate answer. Please try again."
        return "Gagal menghasilkan jawaban. Silakan coba lagi."
