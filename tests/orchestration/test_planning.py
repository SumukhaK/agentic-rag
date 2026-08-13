from unittest.mock import patch

from agentic_rag.orchestration.planning import plan_and_retrieve
from agentic_rag.retrieval.search import SearchCandidate

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
    user_tier="tier-1",
    known_tiers=["tier-1", "tier-2"],
    retrieval_top_k=10,
    rerank_top_k=4,
    max_attempts=5,
)


def _candidate(text="Arsenal drew 1-1."):
    return SearchCandidate(
        relative_path="tier-1/a.txt",
        chunk_index=0,
        text=text,
        access_tier="tier-1",
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
def test_plan_and_retrieve_exhausts_retries_and_reports_insufficient(
    mock_decompose, mock_search, mock_rerank
):
    mock_decompose.return_value = ["Who won it?"]
    mock_search.return_value = []  # never finds anything

    result = plan_and_retrieve(**{**KWARGS, "max_attempts": 3})

    assert result.sufficient is False
    assert result.attempts_used == 3
    assert mock_decompose.call_count == 3
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
