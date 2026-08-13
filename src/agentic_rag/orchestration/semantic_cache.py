import math
from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.embedding.cache import EmbeddingCache, embed_with_cache
from agentic_rag.embedding.ollama_client import embed_texts
from agentic_rag.orchestration.answer import generate_answer
from agentic_rag.orchestration.planning import plan_and_retrieve


@dataclass(frozen=True)
class _CacheEntry:
    query_embedding: list[float]
    user_tier: str
    answer: str


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """In-memory cache from (query meaning, user_tier) to a previously
    generated final answer, so a semantically-similar repeat question can
    skip the full retrieval+generation pipeline.

    Scoped per `user_tier`, not just per query: a cached answer was
    generated from retrieval already filtered to the tier that produced it
    (REQUIREMENTS.md §11/FR3), so serving it to a different tier could leak
    content that tier isn't entitled to, or under-serve one that is - two
    users at different tiers asking near-identical questions must never
    share a cache entry.

    Same lifetime caveats as `EmbeddingCache` (embedding/cache.py), and for
    the same reason - flagged here rather than silently assumed away:
    in-memory only, no persistence across restarts, and unbounded (no
    eviction, no TTL), so a stale answer can outlive the document that
    invalidated it (FR4's near-real-time freshness applies to retrieval,
    not to whatever a cache decided to skip retrieval for).
    """

    def __init__(self) -> None:
        self._entries: list[_CacheEntry] = []

    def get(
        self, query_embedding: list[float], user_tier: str, *, similarity_threshold: float
    ) -> str | None:
        best_similarity = -1.0
        best_answer: str | None = None
        for entry in self._entries:
            if entry.user_tier != user_tier:
                continue
            similarity = _cosine_similarity(query_embedding, entry.query_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = entry.answer
        if best_answer is not None and best_similarity >= similarity_threshold:
            return best_answer
        return None

    def put(self, query_embedding: list[float], user_tier: str, answer: str) -> None:
        self._entries.append(
            _CacheEntry(
                query_embedding=list(query_embedding), user_tier=user_tier, answer=answer
            )
        )


def answer_with_cache(
    query: str,
    user_tier: str,
    *,
    cache: SemanticCache,
    client: QdrantClient,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout_seconds: int,
    sparse_model: str,
    embedding_cache: EmbeddingCache,
    reranker_model: str,
    generation_model: str,
    generation_timeout_seconds: int,
    known_tiers: list[str],
    retrieval_top_k: int,
    rerank_top_k: int,
    max_attempts: int,
    similarity_threshold: float,
) -> str:
    """Answer `query` for `user_tier`, serving a cached answer for a
    semantically-similar past query at the same tier instead of re-running
    retrieval and generation.

    `query` is expected to already be the final, standalone form of the
    question (history already resolved by `rewrite_query()` upstream if
    this is a multi-turn conversation) - caching operates on query meaning,
    which only makes sense once ambiguous references like "it"/"them" have
    already been resolved into an actual, comparable question.

    The embedding used for the cache lookup is `query`'s own dense
    embedding, distinct from (and computed separately from) the embeddings
    `plan_and_retrieve` computes for its decomposed sub-questions - caching
    compares whole-question meaning, decomposition compares per-sub-question
    meaning, and there's no way to derive one from the other without first
    calling `decompose_query` itself.
    """
    query_embedding = embed_with_cache(
        [query],
        model=embedding_model,
        cache=embedding_cache,
        embed_fn=lambda batch: embed_texts(
            batch, model=embedding_model, base_url=ollama_base_url, timeout=embedding_timeout_seconds
        ),
    )[0]

    cached_answer = cache.get(query_embedding, user_tier, similarity_threshold=similarity_threshold)
    if cached_answer is not None:
        return cached_answer

    planning_result = plan_and_retrieve(
        client,
        collection_name,
        query,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        embedding_timeout_seconds=embedding_timeout_seconds,
        sparse_model=sparse_model,
        embedding_cache=embedding_cache,
        reranker_model=reranker_model,
        generation_model=generation_model,
        generation_timeout_seconds=generation_timeout_seconds,
        user_tier=user_tier,
        known_tiers=known_tiers,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        max_attempts=max_attempts,
    )
    answer = generate_answer(
        planning_result,
        query=query,
        model=generation_model,
        base_url=ollama_base_url,
        timeout=generation_timeout_seconds,
    )

    cache.put(query_embedding, user_tier, answer)
    return answer
