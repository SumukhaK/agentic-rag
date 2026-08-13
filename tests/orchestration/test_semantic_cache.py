import pytest

from agentic_rag.orchestration.semantic_cache import SemanticCache


def test_get_returns_none_when_the_cache_is_empty():
    cache = SemanticCache()

    result = cache.get([1.0, 0.0], "tier-1", similarity_threshold=0.95)

    assert result is None


def test_get_returns_the_cached_answer_for_an_identical_embedding():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", "Arsenal won 2-1.")

    result = cache.get([1.0, 0.0], "tier-1", similarity_threshold=0.95)

    assert result == "Arsenal won 2-1."


def test_get_returns_the_cached_answer_for_a_similar_enough_embedding():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", "Arsenal won 2-1.")

    # Cosine similarity between [1, 0] and [0.99, 0.01] is ~0.9999 - well
    # above a 0.95 threshold, simulating a semantically-near-identical
    # rephrasing of the same question.
    result = cache.get([0.99, 0.01], "tier-1", similarity_threshold=0.95)

    assert result == "Arsenal won 2-1."


def test_get_returns_none_below_the_similarity_threshold():
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", "Arsenal won 2-1.")

    # Orthogonal vectors -> cosine similarity 0.0, far below any
    # reasonable threshold - simulates an unrelated question.
    result = cache.get([0.0, 1.0], "tier-1", similarity_threshold=0.95)

    assert result is None


def test_get_never_returns_an_entry_cached_under_a_different_user_tier():
    # A cached answer was generated from retrieval already filtered to the
    # tier that produced it (FR3) - serving it to a different tier could
    # leak content that tier isn't entitled to, or under-serve one that is.
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", "tier-1's answer.")

    result = cache.get([1.0, 0.0], "tier-2", similarity_threshold=0.95)

    assert result is None


def test_get_picks_the_most_similar_entry_among_several():
    cache = SemanticCache()
    cache.put([0.9, 0.1], "tier-1", "weakly similar answer")
    cache.put([0.0, 1.0], "tier-1", "orthogonal answer")
    cache.put([1.0, 0.0], "tier-1", "closest match")

    result = cache.get([1.0, 0.0], "tier-1", similarity_threshold=0.9)

    assert result == "closest match"


def test_get_raises_on_mismatched_embedding_dimensions():
    cache = SemanticCache()
    cache.put([1.0, 0.0, 0.0], "tier-1", "answer")

    with pytest.raises(ValueError):
        cache.get([1.0, 0.0], "tier-1", similarity_threshold=0.95)
