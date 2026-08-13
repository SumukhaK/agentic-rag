from unittest.mock import patch

from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, PlanningResult
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache

KWARGS = dict(
    client=object(),
    collection_name="documents",
    embedding_model="nomic-embed-text",
    ollama_base_url="http://localhost:11434",
    embedding_timeout_seconds=30,
    sparse_model="Qdrant/bm25",
    reranker_model="BAAI/bge-reranker-base",
    generation_model="mistral",
    generation_timeout_seconds=60,
    known_tiers=["tier-1", "tier-2"],
    retrieval_top_k=10,
    rerank_top_k=4,
    max_attempts=5,
    similarity_threshold=0.95,
)


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.orchestration.semantic_cache.embed_texts")
def test_answer_with_cache_runs_the_full_pipeline_on_a_cache_miss(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = PlanningResult(
        sufficient=True, outcomes=[], attempts_used=1, message=None
    )
    mock_generate.return_value = "Arsenal won 2-1. [1]"
    cache = SemanticCache()

    answer = answer_with_cache(
        "Who won?", "tier-1", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    assert answer == "Arsenal won 2-1. [1]"
    mock_plan.assert_called_once()
    mock_generate.assert_called_once()


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.orchestration.semantic_cache.embed_texts")
def test_answer_with_cache_skips_the_pipeline_on_a_cache_hit(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", "Arsenal won 2-1. [1]")

    answer = answer_with_cache(
        "Who won?", "tier-1", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    assert answer == "Arsenal won 2-1. [1]"
    mock_plan.assert_not_called()
    mock_generate.assert_not_called()


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.orchestration.semantic_cache.embed_texts")
def test_answer_with_cache_populates_the_cache_after_a_miss(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = PlanningResult(
        sufficient=True, outcomes=[], attempts_used=1, message=None
    )
    mock_generate.return_value = "Arsenal won 2-1. [1]"
    cache = SemanticCache()

    answer_with_cache(
        "Who won?", "tier-1", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    assert cache.get([1.0, 0.0], "tier-1", similarity_threshold=0.95) == "Arsenal won 2-1. [1]"


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.orchestration.semantic_cache.embed_texts")
def test_answer_with_cache_does_not_cache_across_different_tiers(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    cache = SemanticCache()
    cache.put([1.0, 0.0], "tier-1", "tier-1's cached answer")
    mock_plan.return_value = PlanningResult(
        sufficient=False, outcomes=[], attempts_used=5, message=CANNOT_ANSWER_MESSAGE
    )
    mock_generate.return_value = CANNOT_ANSWER_MESSAGE

    answer = answer_with_cache(
        "Who won?", "tier-2", cache=cache, embedding_cache=EmbeddingCache(), **KWARGS
    )

    assert answer == CANNOT_ANSWER_MESSAGE
    mock_plan.assert_called_once()


@patch("agentic_rag.orchestration.semantic_cache.generate_answer")
@patch("agentic_rag.orchestration.semantic_cache.plan_and_retrieve")
@patch("agentic_rag.orchestration.semantic_cache.embed_texts")
def test_answer_with_cache_passes_query_and_user_tier_through_to_the_pipeline(
    mock_embed, mock_plan, mock_generate
):
    mock_embed.return_value = [[1.0, 0.0]]
    mock_plan.return_value = PlanningResult(
        sufficient=True, outcomes=[], attempts_used=1, message=None
    )
    mock_generate.return_value = "answer [1]"
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
