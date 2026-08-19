import math
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache, embed_query_dense
from agentic_rag.orchestration.answer import AnswerResult, generate_answer
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, plan_and_retrieve


@dataclass(frozen=True)
class _CacheEntry:
    query_embedding: list[float]
    embedding_model: str
    answer: AnswerResult
    cached_at: float


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # Clamp against floating point drift from independently-summed
    # reductions - a vector compared against itself can otherwise compute
    # to e.g. 1.0000000000000002, which would incorrectly fail a
    # similarity_threshold of exactly 1.0.
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class SemanticCache:
    """In-memory cache from (query meaning, user_tier, embedding_model) to a
    previously generated final answer, so a semantically-similar repeat
    question can skip the full retrieval+generation pipeline.

    Scoped per `user_tier`, not just query meaning: a cached answer was
    generated from retrieval already filtered to the tier that produced it
    (REQUIREMENTS.md §11/FR3), so serving it to a different tier could leak
    content that tier isn't entitled to, or under-serve one that is - two
    users at different tiers asking near-identical questions must never
    share a cache entry. Entries are stored per-tier (not a single flat
    list filtered inline) so a lookup only ever scans its own tier's
    entries, not every other tier's too.

    Also scoped per `embedding_model`: entries from a different model can
    have different dimensionality (or simply not be comparable), so an
    entry from a different model is skipped rather than ever being allowed
    to crash or corrupt a lookup for the model actually in use.

    Entries expire after `ttl_seconds` (checked at read time via `get`,
    not proactively swept) - this bounds, but does not eliminate, two real
    staleness risks flagged during self-review rather than solved outright:
    the corpus can change (FR4's near-real-time freshness) after an answer
    is cached, and - since this system's access-tier model is folder-per-
    tier (REQUIREMENTS.md §11) - a document can be *reclassified* to a
    stricter tier by simply moving it, which the cache has no hook to
    detect. An unbounded TTL would let a cached answer keep citing deleted
    or since-restricted content indefinitely; a bounded one caps the
    exposure window instead. `answer_with_cache` also never caches the
    canonical "I do not know" fallback (see its own docstring) so a
    not-yet-ingested document can't be permanently masked by a cached
    negative result.

    Same "no persistence across restarts" caveat as `EmbeddingCache`
    (embedding/cache.py), for the same reason - not solved here.

    Stores the full `AnswerResult` (text + citations), not just the answer
    text - a cache hit used to return a bare string, silently dropping the
    citations computed when the answer was first generated, so a repeat of
    a semantically-similar question would lose FR1's per-source citations
    on every hit even though the original ask had them. Found during
    self-review of the `POST /query` PR (`PROJECT_TRACKER.md`'s Phase 7
    log).
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[_CacheEntry]] = {}

    def get(
        self,
        query_embedding: list[float],
        user_tier: str,
        embedding_model: str,
        *,
        similarity_threshold: float,
        ttl_seconds: float,
        now: float | None = None,
    ) -> AnswerResult | None:
        current_time = time.time() if now is None else now
        best_similarity = -1.0
        best_answer: AnswerResult | None = None
        for entry in self._entries.get(user_tier, []):
            if entry.embedding_model != embedding_model:
                continue
            if current_time - entry.cached_at > ttl_seconds:
                continue
            similarity = _cosine_similarity(query_embedding, entry.query_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = entry.answer
        if best_answer is not None and best_similarity >= similarity_threshold:
            return best_answer
        return None

    def put(
        self,
        query_embedding: list[float],
        user_tier: str,
        embedding_model: str,
        answer: AnswerResult,
        *,
        now: float | None = None,
    ) -> None:
        current_time = time.time() if now is None else now
        entry = _CacheEntry(
            query_embedding=list(query_embedding),
            embedding_model=embedding_model,
            answer=answer,
            cached_at=current_time,
        )
        self._entries.setdefault(user_tier, []).append(entry)


def answer_with_cache(
    query: str,
    user_tier: str,
    *,
    cache: SemanticCache,
    client: QdrantClient,
    collection_name: str,
    embedding_cache: EmbeddingCache,
    known_tiers: list[str],
    settings: Settings,
) -> AnswerResult:
    """Answer `query` for `user_tier`, serving a cached answer for a
    semantically-similar past query at the same tier instead of re-running
    retrieval and generation.

    `query` is expected to already be the final, standalone form of the
    question (history already resolved by `rewrite_query()` upstream if
    this is a multi-turn conversation) - caching operates on query meaning,
    which only makes sense once ambiguous references like "it"/"them" have
    already been resolved into an actual, comparable question.

    `collection_name` and `known_tiers` stay explicit parameters rather
    than being read off `settings` directly, since every real caller
    deliberately points them at a different value than `settings`' own
    defaults would give (`evaluation`/`loadtest` each index into their own
    dedicated Qdrant collection, and the load test validates its
    hardcoded query tiers against its own fixed corpus layout, not
    whatever `settings.access_tiers` happens to be configured to - see
    each caller for the specific reasoning). Every other tunable this
    function needs is read straight off `settings` instead of being
    accepted as its own parameter - self-review of `POST /query` (this
    function's first real caller) found it hand-marshaling 17 individual
    `Settings` fields into keyword arguments at the one call site, where a
    typo'd kwarg name would be a silent `TypeError` at request time, not
    at import time. Accepting `settings` directly moves that marshaling
    into one place (here) instead of every caller.

    The embedding used for the cache lookup is `query`'s own dense
    embedding - distinct from what `plan_and_retrieve` computes for its
    decomposed sub-questions, since caching compares whole-question
    meaning and decomposition compares per-sub-question meaning. In the
    common case where the query is already simple (`decompose_query`
    returns it unchanged as the only sub-question), both end up embedding
    the same string, and the shared `embedding_cache` makes the second
    embedding a cache hit rather than a second Ollama round-trip - but
    this depends on the two texts matching exactly, so it's an incidental
    win when it happens, not something relied upon.

    A cache hit returns the stored answer as-is, without re-running
    `_is_grounded()` (answer.py) against it - that validation only ran
    once, at write time, against the evidence available then. This is a
    known, deliberately unresolved gap alongside the TTL/staleness caveats
    on `SemanticCache` itself, not something this function papers over.

    Only cached when `planning_result.sufficient` AND the generated answer
    itself doesn't contain the canonical fallback phrase (see
    `SemanticCache`'s docstring for why a fallback specifically must not
    be cached). Checking `sufficient` alone isn't enough: it's a coarse,
    retrieval-only signal (planning.py) that can still be `True` for a
    genuinely unanswerable question, and a model can hedge with an answer
    that opens with the fallback phrase but tacks on a citation that
    happens to pass `_is_grounded()`'s in-range check - live-observed, not
    hypothetical. The answer's own content is the more direct signal for
    "this is actually a non-answer" than the signal that led to generating
    it in the first place.

    `settings.generation_temperature` is threaded straight through to
    `generate_answer()` - see that function's docstring for why it's
    required and a separate setting from the judges' `judge_temperature`.
    `settings.decompose_temperature`/`settings.decompose_retry_temperature` are
    threaded straight through to `plan_and_retrieve()` - see its docstring
    (`planning.py`) for why decomposition needs two temperatures, not one,
    unlike every other pinned-temperature call in this codebase.

    Returns the full `AnswerResult` (text + citations) on both the cache-hit
    and cache-miss paths - see `SemanticCache`'s docstring for why a cache
    hit specifically needed fixing to keep returning citations rather than
    just the answer text.
    """
    query_embedding = embed_query_dense(
        query,
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        timeout=settings.embedding_timeout_seconds,
        cache=embedding_cache,
    )

    cached_answer = cache.get(
        query_embedding,
        user_tier,
        settings.embedding_model,
        similarity_threshold=settings.semantic_cache_similarity_threshold,
        ttl_seconds=settings.semantic_cache_ttl_seconds,
    )
    if cached_answer is not None:
        return cached_answer

    planning_result = plan_and_retrieve(
        client,
        collection_name,
        query,
        embedding_model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
        embedding_timeout_seconds=settings.embedding_timeout_seconds,
        sparse_model=settings.sparse_embedding_model,
        embedding_cache=embedding_cache,
        reranker_model=settings.reranker_model,
        generation_model=settings.generation_model,
        generation_timeout_seconds=settings.generation_timeout_seconds,
        decompose_temperature=settings.decompose_temperature,
        decompose_retry_temperature=settings.decompose_retry_temperature,
        user_tier=user_tier,
        known_tiers=known_tiers,
        retrieval_top_k=settings.retrieval_top_k_candidates,
        rerank_top_k=settings.rerank_top_k,
        max_attempts=settings.max_retrieval_attempts,
    )
    answer = generate_answer(
        planning_result,
        query=query,
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        timeout=settings.generation_timeout_seconds,
        temperature=settings.generation_temperature,
    )

    if planning_result.sufficient and CANNOT_ANSWER_MESSAGE not in answer.text:
        cache.put(query_embedding, user_tier, settings.embedding_model, answer)

    return answer
