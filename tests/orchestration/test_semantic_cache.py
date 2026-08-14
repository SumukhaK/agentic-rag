from unittest.mock import patch

import pytest

from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.answer import AnswerResult, Citation
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, PlanningResult
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache

MODEL = "nomic-embed-text"


def _sufficient_result() -> PlanningResult:
    return PlanningResult(sufficient=True, outcomes=[], attempts_used=1, message=None)


def _insufficient_result() -> PlanningResult:
    return PlanningResult(
        sufficient=False, outcomes=[], attempts_used=5, message=CANNOT_ANSWER_MESSAGE
    )


def _answer(text: str, citations: list[Citation] | None = None) -> AnswerResult:
    return AnswerResult(text=text, citations=citations or [])


_CITATION = Citation(number=1, relative_path="tier-1/a.txt", chunk_index=0, access_tier="tier-1")


# --- SemanticCache: get/put primitive -------------------------------------


def test_get_returns_none_when_the_cache_is_empty():
    cache = SemanticCache()

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)

    assert result is None


def test_get_returns_the_cached_answer_for_an_identical_embedding():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("Arsenal won 2-1."))

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)

    assert result == _answer("Arsenal won 2-1.")


def test_get_preserves_citations_for_a_cache_hit():
    # The gap self-review of the POST /query PR found: a cache hit used to
    # return only the bare answer string, discarding the citation metadata
    # computed when the answer was first generated - so a semantically
    # repeated question would lose FR1's citations entirely on the second
    # (cached) ask, even though the first ask had them.
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("Arsenal won 2-1. [1]", [_CITATION]))

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)

    assert result.citations == [_CITATION]


def test_get_returns_the_cached_answer_for_a_similar_enough_embedding():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("Arsenal won 2-1."))

    # Cosine similarity between [1, 0] and [0.99, 0.01] is ~0.9999 - well
    # above a 0.95 threshold, simulating a semantically-near-identical
    # rephrasing of the same question.
    result = cache.get([0.99, 0.01], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)

    assert result == _answer("Arsenal won 2-1.")


def test_get_returns_none_below_the_similarity_threshold():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("Arsenal won 2-1."))

    # Orthogonal vectors -> cosine similarity 0.0, far below any
    # reasonable threshold - simulates an unrelated question.
    result = cache.get([0.0, 1.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)

    assert result is None


def test_get_never_returns_an_entry_cached_under_a_different_user_tier():
    # A cached answer was generated from retrieval already filtered to the
    # tier that produced it (FR3) - serving it to a different tier could
    # leak content that tier isn't entitled to, or under-serve one that is.
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("tier-1's answer."))

    result = cache.get([1.0, 0.0], "tier-2", MODEL, similarity_threshold=0.95, ttl_seconds=300)

    assert result is None


def test_get_picks_the_most_similar_entry_among_several():
    cache = SemanticCache()
    cache.put([0.9, 0.1], "tier-1", MODEL, _answer("weakly similar answer"))
    cache.put([0.0, 1.0], "tier-1", MODEL, _answer("orthogonal answer"))
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("closest match"))

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.9, ttl_seconds=300)

    assert result == _answer("closest match")


def test_get_raises_on_mismatched_embedding_dimensions_for_the_same_model():
    cache = SemanticCache()
    cache.put([1.0, 0.0, 0.0], "tier-1", MODEL, _answer("answer"))

    with pytest.raises(ValueError):
        cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)


def test_get_ignores_entries_cached_under_a_different_embedding_model():
    # Entries embedded by different models can have different
    # dimensionality (or just not be comparable at all) - an entry from a
    # different model must never be treated as a candidate match, and must
    # never crash the lookup for an unrelated model either.
    cache = SemanticCache()
    cache.put([1.0, 0.0, 0.0], "tier-1", "other-model", _answer("wrong-model answer"))

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)

    assert result is None


def test_get_ignores_entries_older_than_the_ttl():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("stale answer"), now=1000.0)

    result = cache.get(
        [1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=60, now=1070.0
    )

    assert result is None


def test_get_returns_entries_still_within_the_ttl():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("fresh answer"), now=1000.0)

    result = cache.get(
        [1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=60, now=1030.0
    )

    assert result == _answer("fresh answer")


def test_cosine_similarity_is_clamped_to_the_valid_range():
    # A vector compared against itself must never be rejected by floating
    # point drift pushing similarity a hair above 1.0 or below the
    # similarity_threshold due to summation-order rounding error.
    cache = SemanticCache()
    vector = [0.1] * 768
    cache.put(vector, "tier-1", MODEL, _answer("self-match answer"))

    result = cache.get(vector, "tier-1", MODEL, similarity_threshold=1.0, ttl_seconds=300)

    assert result == _answer("self-match answer")


# --- answer_with_cache: pipeline composition --------------------------------

KWARGS = dict(
    client=object(),
    collection_name="documents",
    embedding_model=MODEL,
    ollama_base_url="http://localhost:11434",
    embedding_timeout_seconds=30,
    sparse_model="Qdrant/bm25",
    reranker_model="BAAI/bge-reranker-base",
    generation_model="mistral",
    generation_timeout_seconds=60,
    generation_temperature=0.0,
    known_tiers=["tier-1", "tier-2"],
    retrieval_top_k=10,
    rerank_top_k=4,
    max_attempts=5,
    similarity_threshold=0.95,
    ttl_seconds=300,
)


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.embedding.cache.embed_texts")
def test_answer_with_cache_runs_the_full_pipeline_on_a_cache_miss(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = _sufficient_result()
    mock_generate.return_value = _answer("Arsenal won 2-1. [1]", [_CITATION])
    cache = SemanticCache()

    answer = answer_with_cache(
        "Who won?", "tier-1", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    assert answer == _answer("Arsenal won 2-1. [1]", [_CITATION])
    mock_plan.assert_called_once()
    mock_generate.assert_called_once()


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.embedding.cache.embed_texts")
def test_answer_with_cache_skips_the_pipeline_on_a_cache_hit(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("Arsenal won 2-1. [1]", [_CITATION]))

    answer = answer_with_cache(
        "Who won?", "tier-1", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    assert answer == _answer("Arsenal won 2-1. [1]", [_CITATION])
    mock_plan.assert_not_called()
    mock_generate.assert_not_called()


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.embedding.cache.embed_texts")
def test_answer_with_cache_populates_the_cache_after_a_miss(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = _sufficient_result()
    mock_generate.return_value = _answer("Arsenal won 2-1. [1]", [_CITATION])
    cache = SemanticCache()

    answer_with_cache(
        "Who won?", "tier-1", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)
    assert result == _answer("Arsenal won 2-1. [1]", [_CITATION])


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.embedding.cache.embed_texts")
def test_answer_with_cache_does_not_cache_the_canonical_fallback(
    mock_embed, mock_plan, mock_generate
):
    # Caching "I do not know" would create a negative cache that never
    # self-corrects: if the relevant document is ingested moments later
    # (FR4's near-real-time freshness), every semantically-similar repeat
    # of the same question would keep getting the stale fallback instead
    # of reaching the now-correct pipeline.
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = _insufficient_result()
    mock_generate.return_value = _answer(CANNOT_ANSWER_MESSAGE)
    cache = SemanticCache()

    answer_with_cache(
        "Who won?", "tier-1", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)
    assert result is None


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.embedding.cache.embed_texts")
def test_answer_with_cache_does_not_cache_an_answer_containing_the_fallback_even_when_sufficient(
    mock_embed, mock_plan, mock_generate
):
    # Live-observed, not hypothetical: plan_and_retrieve's coarse
    # sufficient=True signal can still misfire (a tiny corpus returns
    # *something* for any query), and the model can hedge with an answer
    # that starts with the canonical fallback phrase but tacks on a
    # trailing, technically-in-range citation - which passes
    # _is_grounded()'s check. Checking planning_result.sufficient alone
    # isn't a strong enough signal; the answer's own content must be
    # checked too, or a non-answer like this gets cached as if stable.
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = _sufficient_result()
    mock_generate.return_value = _answer(
        f" {CANNOT_ANSWER_MESSAGE} [1] does not contain information about that.", [_CITATION]
    )
    cache = SemanticCache()

    answer_with_cache(
        "What is the capital of France?",
        "tier-1",
        cache=cache,
        embedding_cache=EmbeddingCache(),
        **KWARGS,
    )

    result = cache.get([1.0, 0.0], "tier-1", MODEL, similarity_threshold=0.95, ttl_seconds=300)
    assert result is None


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.embedding.cache.embed_texts")
def test_answer_with_cache_does_not_cache_across_different_tiers(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", MODEL, _answer("tier-1's cached answer"))
    mock_plan.return_value = _insufficient_result()
    mock_generate.return_value = _answer(CANNOT_ANSWER_MESSAGE)

    answer = answer_with_cache(
        "Who won?", "tier-2", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    assert answer == _answer(CANNOT_ANSWER_MESSAGE)
    mock_plan.assert_called_once()


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.embedding.cache.embed_texts")
def test_answer_with_cache_passes_query_and_user_tier_through_to_the_pipeline(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = _sufficient_result()
    mock_generate.return_value = _answer("answer [1]", [_CITATION])
    cache = SemanticCache()

    answer_with_cache(
        "Who won the match?",
        "tier-2",
        cache=cache,
        embedding_cache=EmbeddingCache(),
        **KWARGS,
    )

    assert mock_plan.call_args.kwargs["user_tier"] == "tier-2"
    assert mock_plan.call_args.args[2] == "Who won the match?"
    assert mock_generate.call_args.kwargs["query"] == "Who won the match?"
