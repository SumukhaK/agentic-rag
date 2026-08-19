from unittest.mock import patch

import pytest

from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.answer import Citation, generate_answer
from agentic_rag.orchestration.planning import (
    CANNOT_ANSWER_MESSAGE,
    PlanningResult,
    RetrievalOutcome,
)
from agentic_rag.retrieval.search import SearchCandidate
from access_tiers import TIER_EMPLOYEE

KWARGS = dict(model="mistral", base_url="http://localhost:11434", timeout=60, temperature=0.0)


def _candidate(relative_path="employee/a.txt", chunk_index=0, text="Arsenal drew 1-1.", access_tier=TIER_EMPLOYEE):
    return SearchCandidate(
        relative_path=relative_path,
        chunk_index=chunk_index,
        text=text,
        access_tier=access_tier,
        score=1.0,
    )


def _sufficient_result(outcomes: list[RetrievalOutcome]) -> PlanningResult:
    return PlanningResult(
        sufficient=True, outcomes=outcomes, attempts_used=1, message=None
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

    assert answer.text == CANNOT_ANSWER_MESSAGE
    assert answer.citations == []
    mock_generate.assert_not_called()


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_returns_the_generated_text_when_citations_are_valid(mock_generate):
    mock_generate.return_value = "Arsenal drew 1-1 against Chelsea [1]."
    planning_result = _sufficient_result(
        [RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])]
    )

    answer = generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert answer.text == "Arsenal drew 1-1 against Chelsea [1]."
    assert answer.citations == [
        Citation(number=1, relative_path="employee/a.txt", chunk_index=0, access_tier=TIER_EMPLOYEE)
    ]


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_passes_temperature_through_to_generate(mock_generate):
    mock_generate.return_value = "Arsenal drew 1-1 against Chelsea [1]."
    planning_result = _sufficient_result(
        [RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])]
    )

    generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert mock_generate.call_args.kwargs["temperature"] == 0.0


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_accepts_the_canonical_fallback_verbatim_with_no_citation_required(mock_generate):
    mock_generate.return_value = CANNOT_ANSWER_MESSAGE
    planning_result = _sufficient_result(
        [RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])]
    )

    answer = generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert answer.text == CANNOT_ANSWER_MESSAGE
    assert answer.citations == []


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_falls_back_when_the_generated_text_has_no_citation(mock_generate):
    # An answer with zero citations violates §8 rule 1 just as much as a
    # hallucinated one - the model ignored the "cite every claim"
    # instruction, so the answer can't be trusted as grounded.
    mock_generate.return_value = "Arsenal drew 1-1 against Chelsea."
    planning_result = _sufficient_result(
        [RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])]
    )

    answer = generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert answer.text == CANNOT_ANSWER_MESSAGE
    assert answer.citations == []


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_falls_back_when_a_citation_number_is_out_of_range(mock_generate):
    # Only one source was provided; [2] is a hallucinated citation - worse
    # than no citation, since it carries false authority.
    mock_generate.return_value = "Arsenal drew 1-1 against Chelsea [2]."
    planning_result = _sufficient_result(
        [RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])]
    )

    answer = generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert answer.text == CANNOT_ANSWER_MESSAGE
    assert answer.citations == []


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_prompt_includes_query_grounding_rules_and_sources(mock_generate):
    mock_generate.return_value = "Arsenal drew 1-1 against Chelsea [1]."
    planning_result = _sufficient_result(
        [
            RetrievalOutcome(
                sub_question="Who played?",
                candidates=[_candidate(chunk_index=3, text="Arsenal drew 1-1 against Chelsea.")],
            )
        ]
    )

    generate_answer(planning_result, query="Who scored?", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert "Who scored?" in prompt
    assert "Arsenal drew 1-1 against Chelsea." in prompt
    assert "employee/a.txt" in prompt
    assert TIER_EMPLOYEE in prompt
    assert "3" in prompt  # chunk_index - citations must identify the exact chunk, not just the file
    assert CANNOT_ANSWER_MESSAGE in prompt
    assert "[1]" in prompt


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_deduplicates_candidates_shared_across_sub_questions(mock_generate):
    mock_generate.return_value = "Arsenal drew 1-1 against Chelsea [1]."
    shared = _candidate(text="Arsenal drew 1-1 against Chelsea.")
    planning_result = _sufficient_result(
        [
            RetrievalOutcome(sub_question="Who played?", candidates=[shared]),
            RetrievalOutcome(sub_question="What was the score?", candidates=[shared]),
        ]
    )

    generate_answer(planning_result, query="Recap the match", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert prompt.count("Arsenal drew 1-1 against Chelsea.") == 1
    assert "[2]" not in prompt


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_labels_distinct_candidates_with_increasing_citation_numbers(mock_generate):
    mock_generate.return_value = "First chunk [1]. Second chunk [2]."
    planning_result = _sufficient_result(
        [
            RetrievalOutcome(
                sub_question="Who played?",
                candidates=[
                    _candidate(chunk_index=0, text="First chunk."),
                    _candidate(chunk_index=1, text="Second chunk."),
                ],
            )
        ]
    )

    answer = generate_answer(planning_result, query="Recap", **KWARGS)

    prompt = mock_generate.call_args.args[0]
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert answer.citations == [
        Citation(number=1, relative_path="employee/a.txt", chunk_index=0, access_tier=TIER_EMPLOYEE),
        Citation(number=2, relative_path="employee/a.txt", chunk_index=1, access_tier=TIER_EMPLOYEE),
    ]


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_citations_only_include_numbers_actually_cited(mock_generate):
    # Two sources are offered but the model's answer only cites the first -
    # the second was never used as evidence for anything in the final text,
    # so it shouldn't be reported back as a citation either.
    mock_generate.return_value = "First chunk [1]."
    planning_result = _sufficient_result(
        [
            RetrievalOutcome(
                sub_question="Who played?",
                candidates=[
                    _candidate(chunk_index=0, text="First chunk."),
                    _candidate(chunk_index=1, text="Second chunk."),
                ],
            )
        ]
    )

    answer = generate_answer(planning_result, query="Recap", **KWARGS)

    assert answer.citations == [
        Citation(number=1, relative_path="employee/a.txt", chunk_index=0, access_tier=TIER_EMPLOYEE)
    ]


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_returns_the_fallback_without_calling_generate_when_candidates_are_empty(mock_generate):
    # Defense in depth: sufficient=True with no actual candidates shouldn't
    # be reachable via plan_and_retrieve today, but generate_answer
    # shouldn't fire a pointless LLM call over an empty source list either
    # way.
    planning_result = _sufficient_result(
        [RetrievalOutcome(sub_question="Who played?", candidates=[])]
    )

    answer = generate_answer(planning_result, query="Who scored?", **KWARGS)

    assert answer.text == CANNOT_ANSWER_MESSAGE
    assert answer.citations == []
    mock_generate.assert_not_called()


@patch("agentic_rag.orchestration.answer.generate")
def test_generate_answer_propagates_generation_error(mock_generate):
    mock_generate.side_effect = GenerationError("Ollama is unreachable")
    planning_result = _sufficient_result(
        [RetrievalOutcome(sub_question="Who played?", candidates=[_candidate()])]
    )

    with pytest.raises(GenerationError):
        generate_answer(planning_result, query="Who scored?", **KWARGS)
