from unittest.mock import patch

import pytest

from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.foul_language import (
    FoulLanguageCheckResult,
    check_for_foul_language,
)

KWARGS = dict(model="mistral", base_url="http://localhost:11434", timeout=30, temperature=0.0)


@patch("agentic_rag.orchestration.foul_language.generate")
def test_check_for_foul_language_returns_false_for_a_clean_message(mock_generate):
    mock_generate.return_value = "CLEAN"

    result = check_for_foul_language("Who won the Arsenal vs Chelsea match?", **KWARGS)

    assert result == FoulLanguageCheckResult(is_foul=False, raw_judge_response="CLEAN")


@patch("agentic_rag.orchestration.foul_language.generate")
def test_check_for_foul_language_flags_an_abusive_message(mock_generate):
    mock_generate.return_value = "FOUL"

    result = check_for_foul_language("You're a worthless idiot, answer me now!", **KWARGS)

    assert result.is_foul is True


@patch("agentic_rag.orchestration.foul_language.generate")
def test_check_for_foul_language_is_case_insensitive(mock_generate):
    mock_generate.return_value = "foul"
    assert check_for_foul_language("text", **KWARGS).is_foul is True

    mock_generate.return_value = "clean"
    assert check_for_foul_language("text", **KWARGS).is_foul is False


@patch("agentic_rag.orchestration.foul_language.generate")
def test_check_for_foul_language_fails_closed_on_an_ambiguous_response(mock_generate):
    mock_generate.return_value = "I'm not sure how to classify this."

    result = check_for_foul_language("text", **KWARGS)

    assert result.is_foul is True


@patch("agentic_rag.orchestration.foul_language.generate")
def test_check_for_foul_language_propagates_generation_error(mock_generate):
    mock_generate.side_effect = GenerationError("Ollama is unreachable")

    with pytest.raises(GenerationError):
        check_for_foul_language("text", **KWARGS)


@patch("agentic_rag.orchestration.foul_language.generate")
def test_check_for_foul_language_includes_text_between_delimiters_in_the_prompt(mock_generate):
    mock_generate.return_value = "CLEAN"

    check_for_foul_language("Who won the match?", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert "<<<MESSAGE_START>>>" in prompt
    assert "<<<MESSAGE_END>>>" in prompt
    assert "Who won the match?" in prompt


@patch("agentic_rag.orchestration.foul_language.generate")
def test_check_for_foul_language_passes_temperature_through(mock_generate):
    mock_generate.return_value = "CLEAN"

    check_for_foul_language("text", **KWARGS)

    assert mock_generate.call_args.kwargs["temperature"] == 0.0
