"""
Tests for agent orchestrator logic.
"""

import json

import pytest

from src.agent.orchestrator import _build_tool_descriptions_text, _parse_agent_response
from src.agent.state import AgentStep


class TestParseAgentResponse:
    """Tests for JSON parsing of agent LLM output."""

    def test_parse_tool_call_clean_json(self) -> None:
        text = '{"tool": "retrieve", "arguments": {"query": "revenue"}}'
        result = _parse_agent_response(text)
        assert result is not None
        assert result["tool"] == "retrieve"
        assert result["arguments"]["query"] == "revenue"

    def test_parse_tool_call_in_code_block(self) -> None:
        text = '```json\n{"tool": "retrieve", "arguments": {"query": "test"}}\n```'
        result = _parse_agent_response(text)
        assert result is not None
        assert result["tool"] == "retrieve"

    def test_parse_final_answer(self) -> None:
        text = '{"final_answer": "The revenue grew by 15%."}'
        result = _parse_agent_response(text)
        assert result is not None
        assert "final_answer" in result
        assert result["final_answer"] == "The revenue grew by 15%."

    def test_parse_final_answer_in_code_block(self) -> None:
        text = '```json\n{"final_answer": "Answer here"}\n```'
        result = _parse_agent_response(text)
        assert result is not None
        assert result["final_answer"] == "Answer here"

    def test_parse_json_with_surrounding_text(self) -> None:
        text = 'I will search for info.\n{"tool": "retrieve", "arguments": {"query": "data"}}'
        result = _parse_agent_response(text)
        assert result is not None
        assert result["tool"] == "retrieve"

    def test_parse_invalid_json_returns_none(self) -> None:
        text = "This is just plain text without any JSON."
        result = _parse_agent_response(text)
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        result = _parse_agent_response("")
        assert result is None

    def test_parse_tool_call_no_arguments(self) -> None:
        text = '{"tool": "analyze_sources", "arguments": {}}'
        result = _parse_agent_response(text)
        assert result is not None
        assert result["tool"] == "analyze_sources"
        assert result["arguments"] == {}


class TestBuildToolDescriptions:
    """Tests for tool description generation."""

    def test_descriptions_not_empty(self) -> None:
        text = _build_tool_descriptions_text()
        assert len(text) > 100

    def test_descriptions_contain_all_tools(self) -> None:
        text = _build_tool_descriptions_text()
        assert "retrieve" in text
        assert "analyze_sources" in text
        assert "summarize_context" in text
        assert "refine_query" in text
        assert "compare_sources" in text


class TestAgentStepModel:
    """Tests for AgentStep model behavior."""

    def test_step_types_valid(self) -> None:
        for step_type in ["thought", "action", "observation", "final_answer"]:
            step = AgentStep(step_type=step_type, content="test")
            assert step.step_type == step_type

    def test_step_serialization_roundtrip(self) -> None:
        step = AgentStep(
            step_type="action",
            content="Using retrieve tool",
            tool_name="retrieve",
            tool_input={"query": "test"},
            tool_output="Found 3 results",
        )
        data = step.model_dump()
        restored = AgentStep(**data)
        assert restored.step_type == step.step_type
        assert restored.tool_name == step.tool_name
        assert restored.tool_input == step.tool_input
