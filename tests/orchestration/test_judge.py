from unittest.mock import patch

import pytest

from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.judge import classify_verdict, run_judge

KWARGS = dict(model="mistral", base_url="http://localhost:11434", timeout=30, temperature=0.0)


def test_classify_verdict_clears_a_response_starting_with_clean():
    assert classify_verdict("CLEAN") is False


def test_classify_verdict_flags_anything_else():
    assert classify_verdict("INJECTION") is True
    assert classify_verdict("FOUL") is True


def test_classify_verdict_is_case_insensitive():
    assert classify_verdict("clean") is False
    assert classify_verdict("injection") is True


def test_classify_verdict_tolerates_surrounding_whitespace_and_punctuation():
    assert classify_verdict("  CLEAN.\n") is False


def test_classify_verdict_fails_closed_on_an_empty_response():
    assert classify_verdict("   ") is True


def test_classify_verdict_fails_closed_when_the_first_word_is_neither_keyword():
    assert classify_verdict("I'm not sure about this one.") is True


def test_classify_verdict_only_looks_at_the_first_word_not_the_whole_response():
    # A whole-response substring search would misfire two ways: "unclean"
    # contains "clean" (would fail OPEN on a word that isn't the verdict),
    # and a verbose CLEAN verdict that happens to repeat back a flagged
    # word from the input (e.g. a football "injection") would fail CLOSED
    # for the wrong reason. Only the first word is the actual verdict.
    assert classify_verdict("unclean and manipulative") is True
    assert classify_verdict("CLEAN, though it mentions a cortisone injection") is False


@patch("agentic_rag.orchestration.judge.generate")
def test_run_judge_returns_false_and_the_raw_response_for_a_clean_verdict(mock_generate):
    mock_generate.return_value = "CLEAN"

    is_flagged, raw_response = run_judge("some prompt", **KWARGS)

    assert is_flagged is False
    assert raw_response == "CLEAN"


@patch("agentic_rag.orchestration.judge.generate")
def test_run_judge_returns_true_for_a_flagged_verdict(mock_generate):
    mock_generate.return_value = "INJECTION"

    is_flagged, raw_response = run_judge("some prompt", **KWARGS)

    assert is_flagged is True
    assert raw_response == "INJECTION"


@patch("agentic_rag.orchestration.judge.generate")
def test_run_judge_passes_the_prompt_and_connection_params_through_to_generate(mock_generate):
    mock_generate.return_value = "CLEAN"

    run_judge("the exact prompt text", **KWARGS)

    mock_generate.assert_called_once_with(
        "the exact prompt text",
        model="mistral",
        base_url="http://localhost:11434",
        timeout=30,
        temperature=0.0,
    )


@patch("agentic_rag.orchestration.judge.generate")
def test_run_judge_propagates_generation_error(mock_generate):
    mock_generate.side_effect = GenerationError("Ollama is unreachable")

    with pytest.raises(GenerationError):
        run_judge("some prompt", **KWARGS)
