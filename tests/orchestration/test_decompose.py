from unittest.mock import patch

import pytest

from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.decompose import decompose_query

KWARGS = dict(model="mistral", base_url="http://localhost:11434", timeout=60)


@patch("agentic_rag.orchestration.decompose.generate")
def test_decompose_query_returns_a_single_item_list_for_a_simple_question(
    mock_generate,
):
    mock_generate.return_value = "Who won the match?"

    result = decompose_query("Who won the match?", **KWARGS)

    assert result == ["Who won the match?"]


@patch("agentic_rag.orchestration.decompose.generate")
def test_decompose_query_splits_multiple_sub_questions(mock_generate):
    mock_generate.return_value = (
        "Who won the Arsenal vs Chelsea match?\n"
        "How many goals were scored in total?"
    )

    result = decompose_query("Complex question", **KWARGS)

    assert result == [
        "Who won the Arsenal vs Chelsea match?",
        "How many goals were scored in total?",
    ]


@patch("agentic_rag.orchestration.decompose.generate")
def test_decompose_query_strips_numbering_and_bullets(mock_generate):
    mock_generate.return_value = "1. Who won the match?\n2) How many goals?\n- Any red cards?\n* Attendance?"

    result = decompose_query("Complex question", **KWARGS)

    assert result == [
        "Who won the match?",
        "How many goals?",
        "Any red cards?",
        "Attendance?",
    ]


@patch("agentic_rag.orchestration.decompose.generate")
def test_decompose_query_ignores_blank_lines(mock_generate):
    mock_generate.return_value = "Who won the match?\n\n\nHow many goals?"

    result = decompose_query("Complex question", **KWARGS)

    assert result == ["Who won the match?", "How many goals?"]


@patch("agentic_rag.orchestration.decompose.generate")
def test_decompose_query_raises_when_the_llm_returns_nothing_usable(mock_generate):
    mock_generate.return_value = "   \n  \n"

    with pytest.raises(GenerationError):
        decompose_query("Complex question", **KWARGS)


@patch("agentic_rag.orchestration.decompose.generate")
def test_decompose_query_propagates_generation_error(mock_generate):
    mock_generate.side_effect = GenerationError("Ollama is unreachable")

    with pytest.raises(GenerationError):
        decompose_query("Complex question", **KWARGS)
