from unittest.mock import patch

import pytest

from agentic_rag.embedding.ollama_client import EmbeddingError
from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.planning import (
    CANNOT_ANSWER_MESSAGE,
    PlanningResult,
    plan_and_retrieve,
)
from agentic_rag.retrieval.access import UnknownAccessTierError
from agentic_rag.retrieval.rerank import RerankError
from agentic_rag.retrieval.search import SearchCandidate
from tests.access_tiers import TIER_EMPLOYEE, TIER_MANAGER

KWARGS = dict(
    client=object(),
    collection_name="documents",
    query="Who won the Arsenal Chelsea match?",
    embedding_model="nomic-embed-text",
    ollama_base_url="http://localhost:11434",
    embedding_timeout_seconds=30,
    sparse_model="Qdrant/bm25",
    embedding_cache=object(),
    reranker_model="BAAI/bge-reranker-base",
    generation_model="mistral",
    generation_timeout_seconds=60,
    user_tier=TIER_EMPLOYEE,
    known_tiers=[TIER_EMPLOYEE, TIER_MANAGER],
    retrieval_top_k=10,
    rerank_top_k=4,
    max_attempts=5,
    decompose_temperature=0.0,
    decompose_retry_temperature=0.4,
)


def _candidate(text="Arsenal drew 1-1."):
    return SearchCandidate(
        relative_path="employee/a.txt",
        chunk_index=0,
        text=text,
        access_tier=TIER_EMPLOYEE,
        score=1.0,
    )


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_succeeds_on_first_attempt(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?", "How many goals?"]
    mock_search.return_value = [_candidate()]
    mock_rerank.return_value = [_candidate()]

    result = plan_and_retrieve(**KWARGS)

    assert result.sufficient is True
    assert result.attempts_used == 1
    assert [o.sub_question for o in result.outcomes] == ["Who won it?", "How many goals?"]
    assert all(o.candidates for o in result.outcomes)
    assert mock_decompose.call_count == 1
    assert result.message is None


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_retries_when_a_sub_question_has_no_evidence(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.side_effect = [["Who won it?"], ["Who won the match?"]]
    mock_search.side_effect = [[], [_candidate()]]  # empty first, then found
    mock_rerank.return_value = [_candidate()]

    result = plan_and_retrieve(**KWARGS)

    assert result.sufficient is True
    assert result.attempts_used == 2
    assert mock_decompose.call_count == 2


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_uses_decompose_temperature_on_the_first_attempt(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.return_value = [_candidate()]
    mock_rerank.return_value = [_candidate()]

    plan_and_retrieve(**KWARGS)

    assert mock_decompose.call_args.kwargs["temperature"] == 0.0


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_uses_decompose_retry_temperature_on_later_attempts(
    mock_decompose, mock_search, mock_rerank
):
    # A retry only has a chance of succeeding if it explores a genuinely
    # different phrasing - retrying with the same deterministic
    # decomposition would just reproduce the same "insufficient" result.
    mock_decompose.side_effect = [["Who won it?"], ["Who won the match?"]]
    mock_search.side_effect = [[], [_candidate()]]
    mock_rerank.return_value = [_candidate()]

    plan_and_retrieve(**KWARGS)

    first_call, second_call = mock_decompose.call_args_list
    assert first_call.kwargs["temperature"] == 0.0
    assert second_call.kwargs["temperature"] == 0.4


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_uses_decompose_retry_temperature_for_every_attempt_after_the_first(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.return_value = []

    plan_and_retrieve(**{**KWARGS, "max_attempts": 3})

    temperatures = [call.kwargs["temperature"] for call in mock_decompose.call_args_list]
    assert temperatures == [0.0, 0.4, 0.4]


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_exhausts_retries_and_reports_insufficient(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.return_value = []  # never finds anything

    result = plan_and_retrieve(**{**KWARGS, "max_attempts": 3})

    assert result.sufficient is False
    assert result.attempts_used == 3
    assert mock_decompose.call_count == 3
    assert result.message == CANNOT_ANSWER_MESSAGE
    mock_rerank.assert_not_called()


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_skips_rerank_for_a_sub_question_with_no_candidates(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?", "Any red cards?"]
    mock_search.side_effect = [[_candidate()], []]
    mock_rerank.return_value = [_candidate()]

    plan_and_retrieve(**{**KWARGS, "max_attempts": 1})

    # rerank is only worth calling for sub-questions that actually got
    # candidates back - reranking an empty list is pure wasted work.
    assert mock_rerank.call_count == 1


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_respects_max_attempts_of_one(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.return_value = []

    result = plan_and_retrieve(**{**KWARGS, "max_attempts": 1})

    assert result.sufficient is False
    assert result.attempts_used == 1
    assert mock_decompose.call_count == 1


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_retries_after_a_transient_decompose_failure(
    mock_decompose, mock_search, mock_rerank
):
    # A single bad LLM response is exactly the kind of transient failure
    # the retry budget exists to absorb - it must cost one attempt, not
    # abort the whole call.
    mock_decompose.side_effect = [GenerationError("mistral returned nothing usable"), ["Who won it?"]]
    mock_search.return_value = [_candidate()]
    mock_rerank.return_value = [_candidate()]

    result = plan_and_retrieve(**KWARGS)

    assert result.sufficient is True
    assert result.attempts_used == 2
    assert mock_decompose.call_count == 2


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_retries_after_a_transient_search_failure(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.side_effect = [EmbeddingError("Ollama unreachable"), [_candidate()]]
    mock_rerank.return_value = [_candidate()]

    result = plan_and_retrieve(**KWARGS)

    assert result.sufficient is True
    assert result.attempts_used == 2


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_retries_after_a_transient_rerank_failure(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.return_value = [_candidate()]
    mock_rerank.side_effect = [RerankError("cross-encoder failed"), [_candidate()]]

    result = plan_and_retrieve(**KWARGS)

    assert result.sufficient is True
    assert result.attempts_used == 2


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_reports_insufficient_when_every_attempt_fails_transiently(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.side_effect = EmbeddingError("Ollama unreachable")

    result = plan_and_retrieve(**{**KWARGS, "max_attempts": 2})

    assert result.sufficient is False
    assert result.attempts_used == 2
    assert result.outcomes == []
    assert result.message == CANNOT_ANSWER_MESSAGE


@patch("agentic_rag.orchestration.planning.rerank")
@patch("agentic_rag.orchestration.planning.hybrid_search")
@patch("agentic_rag.orchestration.planning.decompose_query")
def test_plan_and_retrieve_does_not_retry_an_unknown_access_tier(
    mock_decompose, mock_search, mock_rerank
):
    # A bad user_tier is a configuration error, not a transient failure -
    # retrying with a fresh decomposition can never fix it, so it must
    # propagate immediately instead of burning the retry budget.
    mock_decompose.return_value = ["Who won it?"]
    mock_search.side_effect = UnknownAccessTierError("tier-9 is not a known tier")

    with pytest.raises(UnknownAccessTierError):
        plan_and_retrieve(**KWARGS)

    assert mock_decompose.call_count == 1


def test_planning_result_rejects_insufficient_without_a_message():
    # message must always be a real fallback string when sufficient=False -
    # a caller (e.g. generate_answer) trusts this to return str, not
    # Optional[str]; nothing but this invariant guarantees that.
    with pytest.raises(ValueError):
        PlanningResult(sufficient=False, outcomes=[], attempts_used=1, message=None)


def test_planning_result_rejects_sufficient_with_a_message():
    with pytest.raises(ValueError):
        PlanningResult(
            sufficient=True, outcomes=[], attempts_used=1, message=CANNOT_ANSWER_MESSAGE
        )
