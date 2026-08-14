from unittest.mock import patch

import pytest

from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.rewrite import ConversationTurn, rewrite_query

KWARGS = dict(model="mistral", base_url="http://localhost:11434", timeout=60, temperature=0.0)


@patch("agentic_rag.orchestration.rewrite.generate")
def test_rewrite_query_returns_query_unchanged_when_no_history(mock_generate):
    result = rewrite_query([], "Who won the match?", **KWARGS)

    assert result == "Who won the match?"
    mock_generate.assert_not_called()


@patch("agentic_rag.orchestration.rewrite.generate")
def test_rewrite_query_sends_history_and_query_to_the_llm(mock_generate):
    mock_generate.return_value = "What was the result of the Arsenal vs Chelsea match?"
    history = [
        ConversationTurn(
            user_query="Tell me about the Arsenal match.",
            assistant_answer="Arsenal played Chelsea last weekend.",
        )
    ]

    rewrite_query(history, "Who won it?", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert "Tell me about the Arsenal match." in prompt
    assert "Arsenal played Chelsea last weekend." in prompt
    assert "Who won it?" in prompt


@patch("agentic_rag.orchestration.rewrite.generate")
def test_rewrite_query_returns_the_llm_response_stripped(mock_generate):
    mock_generate.return_value = "  What was the result of the match?  \n"
    history = [ConversationTurn(user_query="q", assistant_answer="a")]

    result = rewrite_query(history, "Who won it?", **KWARGS)

    assert result == "What was the result of the match?"


@patch("agentic_rag.orchestration.rewrite.generate")
def test_rewrite_query_passes_temperature_through_to_generate(mock_generate):
    mock_generate.return_value = "What was the result of the match?"
    history = [ConversationTurn(user_query="q", assistant_answer="a")]

    rewrite_query(history, "Who won it?", **KWARGS)

    assert mock_generate.call_args.kwargs["temperature"] == 0.0


@patch("agentic_rag.orchestration.rewrite.generate")
def test_rewrite_query_propagates_generation_error(mock_generate):
    mock_generate.side_effect = GenerationError("Ollama is unreachable")
    history = [ConversationTurn(user_query="q", assistant_answer="a")]

    with pytest.raises(GenerationError):
        rewrite_query(history, "Who won it?", **KWARGS)


@patch("agentic_rag.orchestration.rewrite.generate")
def test_rewrite_query_raises_when_the_llm_returns_an_empty_rewrite(mock_generate):
    # An empty/whitespace-only "rewrite" is a worse failure than an
    # exception: nothing would signal it, and it would silently reach
    # retrieval as a query with nothing to search for.
    mock_generate.return_value = "   \n  "
    history = [ConversationTurn(user_query="q", assistant_answer="a")]

    with pytest.raises(GenerationError):
        rewrite_query(history, "Who won it?", **KWARGS)
