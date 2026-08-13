import pytest

from agentic_rag.retrieval.rerank import RerankError, rerank
from agentic_rag.retrieval.search import SearchCandidate

MODEL = "BAAI/bge-reranker-base"


def _candidate(text, relative_path="tier-1/a.txt", chunk_index=0, score=0.0):
    return SearchCandidate(
        relative_path=relative_path,
        chunk_index=chunk_index,
        text=text,
        access_tier="tier-1",
        score=score,
    )


@pytest.fixture(scope="module", autouse=True)
def _require_reranker_model():
    try:
        rerank("warmup", [_candidate("warmup")], model_name=MODEL, top_k=1)
    except RerankError as exc:
        pytest.skip(f"reranker model unavailable: {exc}")


def test_rerank_orders_candidates_by_relevance_to_the_query():
    relevant = _candidate(
        "Arsenal drew 1-1 against Chelsea in a tense London derby.",
        relative_path="tier-1/football.txt",
    )
    irrelevant = _candidate(
        "It rained heavily across the south of England this weekend.",
        relative_path="tier-1/weather.txt",
    )

    results = rerank(
        "Arsenal Chelsea match result", [irrelevant, relevant], model_name=MODEL, top_k=4
    )

    assert [c.relative_path for c in results] == [
        "tier-1/football.txt",
        "tier-1/weather.txt",
    ]


def test_rerank_returns_at_most_top_k_candidates():
    candidates = [_candidate(f"Arsenal match report number {i}.") for i in range(6)]

    results = rerank("Arsenal", candidates, model_name=MODEL, top_k=4)

    assert len(results) == 4


def test_rerank_returns_all_candidates_when_fewer_than_top_k():
    candidates = [_candidate("Arsenal drew 1-1 against Chelsea.")]

    results = rerank("Arsenal", candidates, model_name=MODEL, top_k=4)

    assert len(results) == 1


def test_rerank_returns_empty_list_for_no_candidates():
    assert rerank("Arsenal", [], model_name=MODEL, top_k=4) == []


def test_rerank_updates_score_to_the_reranker_score():
    candidate = _candidate("Arsenal drew 1-1 against Chelsea.", score=0.5)

    results = rerank("Arsenal Chelsea", [candidate], model_name=MODEL, top_k=4)

    # The reranker's own relevance score replaces the fused-search score -
    # it's a more accurate relevance signal for the chunks actually sent
    # to generation.
    assert results[0].score != 0.5


def test_rerank_raises_rerank_error_for_an_unknown_model():
    with pytest.raises(RerankError):
        rerank("Arsenal", [_candidate("Arsenal drew 1-1.")], model_name="not-a-real-model", top_k=4)
