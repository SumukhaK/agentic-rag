from unittest.mock import patch

import pytest

from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.answer import generate_answer
from agentic_rag.orchestration.planning import (
    CANNOT_ANSWER_MESSAGE,
    PlanningResult,
    RetrievalOutcome,
)
from agentic_rag.retrieval.search import SearchCandidate

KWARGS = dict(model="mistral", base_url="http://localhost:11434", timeout=60)


def _candidate(relative_path="tier-1/a.txt", chunk_index=0, text="Arsenal drew 1-1.", access_tier="tier-1"):
    return SearchCandidate(
        relative_path=relative_path,
        chunk_index=chunk_index,
        text=text,
        access_tier=access_tier,
        score=1.0,
    )


def test_generate_answer_returns_the_fallback_message_without_calling_generate_when_insufficient():
    planning_result = PlanningResult(
        sufficient=False,
        outcomes=[],
        attempts_used=5,
        message=CANNOT_ANSWER_MESSAGE,
    )

    with patch("agentic_rag.orchestration.answer.generate") as mock_generate:
        answer = generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert answer == CANNOT_ANSWER_MESSAGE
    mock_generate.assert_not_called()


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_returns_the_generated_text(mock_generate):
    mock_generate.return_value = "Arsenal drew 1-1 against Chelsea [1]."
    planning_result = PlanningResult(
        sufficient=True,
        outcomes=[
            RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])
        ],
        attempts_used=1,
        message=None,
    )

    answer = generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert answer == "Arsenal drew 1-1 against Chelsea [1]."


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_prompt_includes_query_grounding_rules_and_sources(mock_generate):
    mock_generate.return_value = "answer"
    planning_result = PlanningResult(
        sufficient=True,
        outcomes=[
            RetrievalOutcome(
                sub_question="Who played?",
                candidates=[_candidate(text="Arsenal drew 1-1 against Chelsea.")],
            )
        ],
        attempts_used=1,
        message=None,
    )

    generate_answer(planning_result, query="Who scored?", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert "Who scored?" in prompt
    assert "Arsenal drew 1-1 against Chelsea." in prompt
    assert "tier-1/a.txt" in prompt
    assert "tier-1" in prompt
    assert CANNOT_ANSWER_MESSAGE in prompt
    assert "[1]" in prompt


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_deduplicates_candidates_shared_across_sub_questions(mock_generate):
    mock_generate.return_value = "answer"
    shared = _candidate(text="Arsenal drew 1-1 against Chelsea.")
    planning_result = PlanningResult(
        sufficient=True,
        outcomes=[
            RetrievalOutcome(sub_question="Who played?", candidates=[shared]),
            RetrievalOutcome(sub_question="What was the score?", candidates=[shared]),
        ],
        attempts_used=1,
        message=None,
    )

    generate_answer(planning_result, query="Recap the match", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert prompt.count("Arsenal drew 1-1 against Chelsea.") == 1
    assert "[2]" not in prompt


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_labels_distinct_candidates_with_increasing_citation_numbers(mock_generate):
    mock_generate.return_value = "answer"
    planning_result = PlanningResult(
        sufficient=True,
        outcomes=[
            RetrievalOutcome(
                sub_question="Who played?",
                candidates=[
                    _candidate(chunk_index=0, text="First chunk."),
                    _candidate(chunk_index=1, text="Second chunk."),
                ],
            )
        ],
        attempts_used=1,
        message=None,
    )

    generate_answer(planning_result, query="Recap", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert "[1]" in prompt
    assert "[2]" in prompt


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_propagates_generation_error(mock_generate):
    mock_generate.side_effect = GenerationError("Ollama is unreachable")
    planning_result = PlanningResult(
        sufficient=True,
        outcomes=[
            RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])
        ],
        attempts_used=1,
        message=None,
    )

    with pytest.raises(GenerationError):
        generate_answer(planning_result, query="Who scored?", **KWARGS)
