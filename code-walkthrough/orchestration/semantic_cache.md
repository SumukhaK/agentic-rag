# `orchestration/semantic_cache.py`

**Purpose:** This file implements a semantic cache — a cache that recognizes when a *new* question means roughly the same thing as a *past* question, even if the wording is different, and reuses the previously computed answer instead of re-running the entire (expensive) retrieval-plus-generation pipeline. "Semantic" here means it compares questions by their meaning (via embeddings and cosine similarity — a measure of how closely two numeric vectors point in the same direction) rather than requiring an exact text match. The file is careful about two correctness pitfalls that would otherwise make caching unsafe: it never lets one user's cached answer leak to a user with a different access permission level, and it never caches the "I don't know" fallback answer, since that would otherwise permanently mask a document that later gets added to the index. It also wires together the full pipeline — cache lookup, then planning/retrieval, then answer generation — into one convenient entry point, `answer_with_cache`.

## Line-by-line walkthrough

### Lines 1-10 — Imports
```python
import math
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache, embed_query_dense
from agentic_rag.orchestration.answer import AnswerResult, generate_answer
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, plan_and_retrieve
```
- `import math` — used below for `math.sqrt()` in the cosine similarity calculation.
- `import time` — used to get the current wall-clock time, for recording when a cache entry was written and for checking whether it has expired.
- `from dataclasses import dataclass` — used to define the internal cache-entry container.
- `from qdrant_client import QdrantClient` — the type of the Qdrant vector database client, passed through to the retrieval step.
- `from agentic_rag.config import Settings` — the application's centralized configuration object; this module reads many of its tunables (thresholds, timeouts, model names) directly from a `Settings` instance rather than accepting each one as a separate parameter.
- `from agentic_rag.embedding.cache import EmbeddingCache, embed_query_dense` — `EmbeddingCache` is a cache of previously computed text embeddings; `embed_query_dense` computes a dense (semantic, meaning-based) embedding for a query string, using that cache to avoid recomputation.
- `from agentic_rag.orchestration.answer import AnswerResult, generate_answer` — `AnswerResult` is the type holding a generated answer's text plus its citations; `generate_answer` is the function that actually produces an answer from retrieved evidence.
- `from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, plan_and_retrieve` — `plan_and_retrieve` is the planner function documented in `planning.py` that decomposes a query, retrieves evidence, and decides if it's sufficient; `CANNOT_ANSWER_MESSAGE` is the shared "I don't know" fallback string defined there, reused here to check whether a generated answer is actually just that fallback.

### Lines 13-18 — `_CacheEntry`
```python
@dataclass(frozen=True)
class _CacheEntry:
    query_embedding: list[float]
    embedding_model: str
    answer: AnswerResult
    cached_at: float
```
- A private (leading underscore), immutable container representing one stored cache entry: the embedding of the query that produced this answer (`query_embedding`), which embedding model produced it (`embedding_model` — important because different models produce incomparable vectors), the actual stored `answer` (the full result, not just text), and `cached_at`, the timestamp (in seconds, from `time.time()`) recording when it was written, used later to check expiry.

### Lines 21-33 — `_cosine_similarity`
```python
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
```
- Computes the cosine similarity between two embedding vectors `a` and `b` — a number between -1 and 1 that measures how similar their directions are, used here as a proxy for "how similar in meaning are these two queries." A value near 1 means very similar meaning; near 0 means unrelated; near -1 means opposite (rare in practice for text embeddings).
- `if len(a) != len(b): raise ValueError(...)` — vectors of different lengths (dimensions) can't be meaningfully compared; this guards against a bug (e.g. accidentally comparing embeddings from two different models) with a clear error rather than a confusing crash or silently wrong number deeper in the math.
- `dot = sum(x * y for x, y in zip(a, b))` — computes the dot product: pairs up corresponding elements of `a` and `b` (via `zip`), multiplies each pair, and sums the results.
- `norm_a = math.sqrt(sum(x * x for x in a))` and `norm_b = math.sqrt(sum(y * y for y in b))` — compute the Euclidean length ("norm," i.e. magnitude) of each vector: the square root of the sum of its squared components.
- `if norm_a == 0 or norm_b == 0: return 0.0` — guards against division by zero: a zero-length vector (all-zero embedding, which shouldn't normally happen but is defensively handled) would make the similarity formula's denominator zero; treating that case as "0 similarity" avoids a crash.
- The comment before the final line explains a subtle floating-point issue: because the dot product and the two norms are each computed via independent summations, tiny rounding errors can accumulate such that comparing an embedding vector against an exact copy of itself computes to something like `1.0000000000000002` instead of exactly `1.0` — a value technically greater than 1, which is mathematically impossible for a true cosine similarity and could wrongly fail a threshold check set to exactly `1.0`.
- `return max(-1.0, min(1.0, dot / (norm_a * norm_b)))` — computes the actual cosine similarity (dot product divided by the product of the two magnitudes), then clamps the result into the valid `[-1.0, 1.0]` range using nested `min`/`max` calls, protecting against the floating-point drift described above.

### Lines 36-79 — `SemanticCache` class docstring
```python
class SemanticCache:
    """In-memory cache from (query meaning, user_tier, embedding_model) to a
    previously generated final answer, so a semantically-similar repeat
    question can skip the full retrieval+generation pipeline.
    ...
    """
```
- The docstring for the whole class lays out its data model and several deliberate safety decisions:
  - It's explicitly an **in-memory** cache (data lives only in this Python process's memory, not on disk or in a database) keyed conceptually by the combination of query meaning, `user_tier`, and `embedding_model`.
  - **Scoped per `user_tier`:** this is explained as a hard requirement, not just an optimization detail — a cached answer was generated from retrieval that was already filtered to whatever tier produced it, so serving that same cached answer to a user at a *different* tier could either leak content that tier shouldn't see, or wrongly under-serve a user who's actually entitled to more. Two users at different tiers asking near-identical questions must never share an entry. Practically, this is implemented by storing entries in separate per-tier buckets (a dictionary keyed by tier) rather than one flat list that gets filtered on every lookup — so a lookup only ever has to scan its own tier's entries, not everyone else's too.
  - **Scoped per `embedding_model`:** entries from a different embedding model might have a different vector dimensionality, or simply not be numerically comparable even if the dimensions happen to match. An entry from the "wrong" model is simply skipped during lookup rather than risking a crash or a nonsensical similarity score.
  - **TTL (time-to-live) expiry, checked lazily:** entries expire after `ttl_seconds`, but that expiry is only checked when a lookup actually happens (at read time, inside `get`), not proactively swept/removed by a background process. The docstring is candid that this only *bounds*, not eliminates, two real staleness risks noted during self-review: the underlying document corpus can change after an answer was cached (the system is meant to reflect near-real-time updates), and — because this system's access-tier model works by literally moving documents between folders — a document could be reclassified to a stricter tier after being cached, and the cache has no way to detect that happened. An unbounded TTL would let a stale cached answer keep citing deleted or now-restricted content indefinitely; a bounded TTL at least caps how long that exposure window can last. It also notes that `answer_with_cache` (below) additionally never caches the canonical "I don't know" fallback specifically so a document that gets indexed *after* a first failed attempt isn't permanently masked by a stale cached negative answer.
  - Notes the cache has the same "doesn't survive a process restart" limitation as the separate `EmbeddingCache`, for the same underlying reason (it's in-memory), and that this is a known, accepted limitation rather than something this file tries to solve.
  - Explains that the cache stores the **full** `AnswerResult` (both the answer text and its citations), not just a plain string — this is called out because an earlier version apparently returned just the bare answer text on a cache hit, silently dropping the citations that were computed when the answer was first generated, so a semantically-similar repeat question would lose its citation information even though the original answer had it. This was found and fixed during a self-review of a related pull request.

### Lines 81-82 — `__init__`
```python
    def __init__(self) -> None:
        self._entries: dict[str, list[_CacheEntry]] = {}
```
- Initializes the cache's internal storage: a dictionary mapping each `user_tier` string to a list of `_CacheEntry` objects for that tier. Starts empty; entries are added via `put()`.

### Lines 84-108 — `get`: looking up a cached answer
```python
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
```
- Takes the new query's `query_embedding`, the requester's `user_tier`, the `embedding_model` used to compute it, a `similarity_threshold` (how similar a past query must be to count as a match) and `ttl_seconds` (how old an entry is allowed to be before it's considered expired). `now` is an optional override for the current time — defaulting to real wall-clock time via `time.time()`, but overridable (useful for deterministic testing without depending on actual elapsed time).
- `current_time = time.time() if now is None else now` — picks the real current time unless a specific `now` was passed in for testing purposes.
- `best_similarity = -1.0` and `best_answer: AnswerResult | None = None` — initialize "best match so far" trackers; -1.0 is the lowest possible cosine similarity, guaranteeing any real candidate will be considered better.
- `for entry in self._entries.get(user_tier, []):` — iterates only over entries stored under this exact `user_tier` (using `.get(user_tier, [])` so a tier with no entries yet just yields an empty list instead of raising a `KeyError`) — this is what makes the per-tier scoping efficient, since entries from other tiers are never even looked at.
- `if entry.embedding_model != embedding_model: continue` — skips any entry computed with a different embedding model, since its vector wouldn't be comparable (or even the same dimensionality).
- `if current_time - entry.cached_at > ttl_seconds: continue` — skips any entry whose age (current time minus when it was cached) exceeds the allowed time-to-live; this is the lazy expiry check described in the class docstring — expired entries are simply ignored on lookup, not proactively deleted.
- `similarity = _cosine_similarity(query_embedding, entry.query_embedding)` — computes how similar, in meaning, the new query is to this stored entry's original query.
- `if similarity > best_similarity: best_similarity = similarity; best_answer = entry.answer` — keeps track of the single best (most similar) match found so far as the loop scans through every still-valid entry for this tier.
- `if best_answer is not None and best_similarity >= similarity_threshold: return best_answer` — after checking every entry, only returns a cache hit if the best match found actually clears the caller's similarity threshold — a merely "closest available" entry that's still too dissimilar is not good enough to reuse.
- `return None` — otherwise, signals a cache miss, telling the caller to actually run the full pipeline.

### Lines 110-126 — `put`: storing a new answer
```python
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
```
- Adds a new entry to the cache. Takes the same identifying info as `get` (the query embedding, tier, model) plus the actual `answer` to store, and the same testable `now` override.
- `current_time = time.time() if now is None else now` — same pattern as in `get`, for the timestamp to store.
- `entry = _CacheEntry(query_embedding=list(query_embedding), embedding_model=embedding_model, answer=answer, cached_at=current_time)` — builds the immutable entry object. `list(query_embedding)` makes a defensive copy of the embedding list, so that if the caller later mutates the list object they originally passed in, it doesn't silently corrupt what's stored in the cache.
- `self._entries.setdefault(user_tier, []).append(entry)` — `setdefault` looks up the list of entries for this `user_tier`, creating and inserting a new empty list first if this is the first entry ever stored for that tier, then appends the new entry to it.

### Lines 129-139 — `answer_with_cache` signature
```python
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
```
- The main, high-level entry point for answering a query with caching. Takes the `query` text and `user_tier`, along with the `cache` (the `SemanticCache` instance to use), Qdrant `client`/`collection_name`, an `embedding_cache` for embedding reuse, `known_tiers`, and the application's `settings` object.

### Lines 140-207 — Docstring: design rationale for the wiring
The docstring explains several deliberate choices in how this function is put together:
- `query` is expected to already be the final, standalone form of the question — any conversation history (like a pronoun such as "it" referring to something mentioned earlier) is assumed to have already been resolved by an upstream `rewrite_query()` step. This matters because meaning-based caching only makes sense once the question is a genuine, self-contained question — comparing two ambiguous fragments wouldn't be meaningful.
- `collection_name` and `known_tiers` are kept as **explicit parameters** rather than being read directly off `settings` like most other tunables, because the docstring notes every real caller of this function actually wants a *different* value than whatever `settings` would default to (an evaluation harness and a load-test harness each use their own dedicated Qdrant collection and their own fixed set of test tiers, not the production defaults). Every *other* tunable this function needs, by contrast, is read straight from `settings` rather than being its own parameter — the docstring explains this was a direct fix for a discovered problem: an earlier version of the `POST /query` API endpoint (the function's first real caller) was hand-copying roughly 17 individual `Settings` fields into keyword arguments at its one call site, where a typo in a keyword argument name would only be caught as a confusing runtime `TypeError` when a request actually came in, not earlier at import time. Accepting the whole `settings` object here moves all that marshaling into one single place instead of repeating it (and its risk of typos) at every call site.
- The embedding used for the cache lookup is computed from the **whole, original `query`** — distinct from the separate embeddings `plan_and_retrieve` computes for each individual decomposed sub-question, because caching is comparing "does this whole question mean the same thing as a past whole question," while decomposition is comparing per-sub-question meaning during retrieval — different granularities serving different purposes. In the common case where a query is already simple enough that decomposition returns it unchanged as its only sub-question, both processes end up embedding the literal same string, and since they share the same underlying `embedding_cache`, the second computation becomes a cheap cache hit rather than a second real call to the embedding model — but the docstring is careful to call this an incidental bonus that only happens when the two texts happen to match exactly, not something the design actually depends on.
- A cache hit returns the previously stored answer exactly as it was, **without** re-running the `_is_grounded()` check (from `answer.py`, which validates that citation numbers are in range) against it — that validation only ever ran once, back when the answer was first generated, against whatever evidence existed at that time. This is explicitly documented as a known, deliberately-unresolved gap, grouped together with the TTL/staleness caveats already noted on `SemanticCache` itself, rather than something this function silently glosses over.
- An answer is only ever cached (written via `cache.put`) when **both** `planning_result.sufficient` is true **and** the generated answer's own text doesn't contain `CANNOT_ANSWER_MESSAGE`. The docstring explains why checking `sufficient` alone wouldn't be enough: it's only a coarse, retrieval-only signal from the planner (some evidence found vs. none) that can still end up `True` for a question that's genuinely unanswerable in practice, and — a case the docstring says was actually observed live, not just theorized — the generation model can produce an answer that *starts* with the fallback wording but still tacks on a citation number that happens to be numerically in-range and therefore passes `_is_grounded()`'s check anyway. Because of that, checking the *answer's own content* for the fallback phrase is a more direct, reliable signal for "this is actually a non-answer" than trusting the upstream signal (`sufficient`) that led to generating it in the first place.
- `settings.generation_temperature` and the two decompose temperatures (`settings.decompose_temperature` / `settings.decompose_retry_temperature`) are simply passed straight through to `generate_answer()` and `plan_and_retrieve()` respectively — the docstring points to those functions' own docstrings (in `answer.py` and `planning.py`) for the actual reasoning behind why these need to be separate, carefully chosen settings rather than one shared value.
- The function returns the full `AnswerResult` (text plus citations) on **both** the cache-hit and cache-miss code paths — consistent with, and referencing, the fix described in `SemanticCache`'s own docstring about not silently dropping citations on a cache hit.

### Lines 208-214 — Computing the query's embedding
```python
    query_embedding = embed_query_dense(
        query,
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        timeout=settings.embedding_timeout_seconds,
        cache=embedding_cache,
    )
```
- Computes the dense (semantic) embedding vector for the whole, original `query`, using the embedding model/server/timeout settings taken from `settings`, and passing in the shared `embedding_cache` so a repeated exact-text embedding computation can be skipped if it was already done recently.

### Lines 216-224 — Checking the cache first
```python
    cached_answer = cache.get(
        query_embedding,
        user_tier,
        settings.embedding_model,
        similarity_threshold=settings.semantic_cache_similarity_threshold,
        ttl_seconds=settings.semantic_cache_ttl_seconds,
    )
    if cached_answer is not None:
        return cached_answer
```
- Looks up the cache for this query's embedding, scoped to the given `user_tier` and `embedding_model`, using the similarity threshold and TTL configured in `settings`.
- `if cached_answer is not None: return cached_answer` — on a cache hit, returns immediately, entirely skipping retrieval and generation — this is the whole point of the cache, saving the cost of the full pipeline for a semantically-similar repeat question.

### Lines 226-245 — Cache miss: running the planner
```python
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
```
- Only reached if the cache lookup missed. Calls the `plan_and_retrieve` function documented in `planning.py`, passing the Qdrant client/collection, the original `query`, and every tunable setting the planner needs — nearly all pulled directly from `settings` (embedding model, Ollama server address, timeouts, sparse embedding model, reranker model, generation model, both decompose temperatures, retrieval/rerank top-k limits, and the max retry attempts), plus the explicitly-passed `user_tier` and `known_tiers` for the reasons explained in the docstring above. The result, `planning_result`, tells this function whether enough evidence was found and what that evidence is.

### Lines 246-253 — Generating the answer
```python
    answer = generate_answer(
        planning_result,
        query=query,
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        timeout=settings.generation_timeout_seconds,
        temperature=settings.generation_temperature,
    )
```
- Passes the `planning_result` (whether retrieval was sufficient, plus whatever evidence was found) along with the original `query` and the generation model/server/timeout/temperature settings into `generate_answer()`, which produces the actual answer text and citations — either grounded in the retrieved evidence, or the canonical fallback message if `planning_result.sufficient` was `False`.

### Lines 255-258 — Deciding whether to cache, and returning
```python
    if planning_result.sufficient and CANNOT_ANSWER_MESSAGE not in answer.text:
        cache.put(query_embedding, user_tier, settings.embedding_model, answer)

    return answer
```
- `if planning_result.sufficient and CANNOT_ANSWER_MESSAGE not in answer.text:` — implements exactly the two-part caching condition explained in the docstring: only store the answer if the planner considered retrieval sufficient *and* the generated answer's actual text doesn't contain the fallback phrase anywhere — this double check exists because either signal alone was found to be insufficiently reliable on its own (see the docstring discussion above about the live-observed case of a hedged answer passing the grounding check despite starting with the fallback wording).
- `cache.put(query_embedding, user_tier, settings.embedding_model, answer)` — if both conditions hold, stores the newly generated answer in the cache under this query's embedding, tier, and embedding model, so a future semantically-similar question at the same tier can reuse it.
- `return answer` — returns the freshly generated `AnswerResult` to the caller, regardless of whether it ended up being cached — this is the cache-miss path's return value, matching the cache-hit path's return type (`AnswerResult`) exactly, as the docstring emphasizes.
