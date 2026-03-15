"""
Agent state models for Agentic RAG.

Defines the data structures for agent reasoning steps and results.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool invocation by the agent."""

    name: str = Field(description="Tool name (retrieve, analyze, summarize, refine_query, compare)")
    arguments: dict = Field(default_factory=dict, description="Tool input arguments")


class AgentStep(BaseModel):
    """A single step in the agent's reasoning process."""

    step_type: Literal["thought", "action", "observation", "final_answer"]
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: str | None = None


class AgentResult(BaseModel):
    """Final result from the agent orchestrator."""

    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    model_used: str = ""
    grounded: bool = True
    grounding_score: float = Field(default=1.0, ge=0, le=1)
    rewritten_query: str | None = None
