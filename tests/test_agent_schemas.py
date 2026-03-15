"""
Tests for agent state models and schemas.
"""

from src.agent.state import AgentResult, AgentStep, ToolCall
from src.api.schemas import AgentStepResponse, QueryRequest, QueryResponse


class TestAgentStateModels:
    """Tests for Pydantic models in agent/state.py"""

    def test_tool_call_creation(self) -> None:
        tc = ToolCall(name="retrieve", arguments={"query": "test"})
        assert tc.name == "retrieve"
        assert tc.arguments == {"query": "test"}

    def test_tool_call_default_arguments(self) -> None:
        tc = ToolCall(name="analyze_sources")
        assert tc.arguments == {}

    def test_agent_step_thought(self) -> None:
        step = AgentStep(step_type="thought", content="I need to search for...")
        assert step.step_type == "thought"
        assert step.tool_name is None

    def test_agent_step_action(self) -> None:
        step = AgentStep(
            step_type="action",
            content="Using retrieve",
            tool_name="retrieve",
            tool_input={"query": "revenue growth"},
        )
        assert step.tool_name == "retrieve"
        assert step.tool_input == {"query": "revenue growth"}

    def test_agent_step_observation(self) -> None:
        step = AgentStep(
            step_type="observation",
            content="Found 5 relevant chunks",
            tool_output="chunk data...",
        )
        assert step.step_type == "observation"
        assert step.tool_output is not None

    def test_agent_step_final_answer(self) -> None:
        step = AgentStep(step_type="final_answer", content="The revenue grew by 15%.")
        assert step.step_type == "final_answer"

    def test_agent_result_defaults(self) -> None:
        result = AgentResult(answer="Test answer")
        assert result.answer == "Test answer"
        assert result.steps == []
        assert result.sources == []
        assert result.iterations == 0
        assert result.grounded is True
        assert result.grounding_score == 1.0

    def test_agent_result_with_steps(self) -> None:
        steps = [
            AgentStep(step_type="thought", content="Thinking..."),
            AgentStep(step_type="action", content="Searching", tool_name="retrieve"),
            AgentStep(step_type="observation", content="Found data"),
            AgentStep(step_type="final_answer", content="Answer here"),
        ]
        result = AgentResult(
            answer="Answer here",
            steps=steps,
            iterations=2,
            model_used="llama-3.3-70b-versatile",
            grounded=True,
            grounding_score=0.85,
        )
        assert len(result.steps) == 4
        assert result.iterations == 2
        assert result.model_used == "llama-3.3-70b-versatile"

    def test_agent_result_serialization(self) -> None:
        result = AgentResult(answer="Test", iterations=1, model_used="test-model")
        data = result.model_dump()
        assert data["answer"] == "Test"
        assert data["iterations"] == 1
        assert isinstance(data["steps"], list)


class TestAgentSchemas:
    """Tests for agent-related API schemas."""

    def test_query_request_agent_mode_default_false(self) -> None:
        req = QueryRequest(question="What is this about?")
        assert req.agent_mode is False

    def test_query_request_agent_mode_true(self) -> None:
        req = QueryRequest(question="What is this?", agent_mode=True)
        assert req.agent_mode is True

    def test_query_request_max_agent_steps_default(self) -> None:
        req = QueryRequest(question="Test?")
        assert req.max_agent_steps == 5

    def test_query_request_max_agent_steps_custom(self) -> None:
        req = QueryRequest(question="Test?", max_agent_steps=3)
        assert req.max_agent_steps == 3

    def test_agent_step_response_serialization(self) -> None:
        step = AgentStepResponse(
            step_type="thought",
            content="Analyzing the question...",
            tool_name=None,
        )
        data = step.model_dump()
        assert data["step_type"] == "thought"
        assert data["content"] == "Analyzing the question..."
        assert data["tool_name"] is None

    def test_query_response_with_agent_fields(self) -> None:
        steps = [
            AgentStepResponse(step_type="thought", content="Thinking..."),
            AgentStepResponse(step_type="action", content="Searching", tool_name="retrieve"),
        ]
        resp = QueryResponse(
            answer="Test answer",
            sources=[],
            model_used="test-model",
            agent_steps=steps,
            agent_iterations=2,
        )
        assert resp.agent_steps is not None
        assert len(resp.agent_steps) == 2
        assert resp.agent_iterations == 2

    def test_query_response_without_agent_fields(self) -> None:
        resp = QueryResponse(
            answer="Test answer",
            sources=[],
            model_used="test-model",
        )
        assert resp.agent_steps is None
        assert resp.agent_iterations is None
