# `retrieval/search.py`

**Purpose:** This file is the heart of the retrieval step in the RAG (retrieval-augmented generation) pipeline: it takes a user's text query and turns it into a ranked list of relevant chunks of text pulled from the Qdrant vector database, while making sure the user only ever sees chunks they're actually allowed to access. It does this with "hybrid search" — running two different kinds of search at once (a "dense" search, which understands semantic meaning via embeddings, and a "sparse" search, which is closer to traditional keyword matching) and then combining ("fusing") their results into one ranking using a technique called RRF (Reciprocal Rank Fusion). This file is where those two search strategies, the access-control filtering from `access.py`, and Qdrant's native fusion machinery all come together into a single function that the rest of the system calls whenever it needs to find relevant context for a query.

## Line-by-line walkthrough

### Lines 1-17 — Imports
```python
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    Prefetch,
)

from agentic_rag.embedding.cache import EmbeddingCache, embed_query_dense, embed_with_cache
from agentic_rag.embedding.sparse_client import embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from agentic_rag.retrieval.access import allowed_tiers_for
```
- `from concurrent.futures import ThreadPoolExecutor` — imports a tool for running two pieces of work on separate threads at the same time (used below to compute the dense and sparse query embeddings concurrently instead of one after another).
- `from dataclasses import dataclass` — imports the decorator used to define `SearchCandidate` as a simple, structured data container below.
- `from qdrant_client import QdrantClient` — imports the client class used to talk to the Qdrant vector database; a `QdrantClient` instance is passed into this file's main function by the caller.
- `from qdrant_client.models import (FieldCondition, Filter, Fusion, FusionQuery, MatchAny, Prefetch)` — imports several Qdrant-specific request-building types: `FieldCondition` and `MatchAny` are used to build a filter based on a payload field's value; `Filter` wraps those conditions together; `Prefetch` describes one "leg" of a multi-stage search (e.g., the dense search or the sparse search) to run before fusion; `FusionQuery` and `Fusion` specify how to combine the results of multiple prefetch legs into one ranking (here, using the RRF algorithm).
- `from agentic_rag.embedding.cache import EmbeddingCache, embed_query_dense, embed_with_cache` — imports the embedding cache type and two helper functions: `embed_query_dense` computes a dense (semantic) embedding vector for a query via Ollama, and `embed_with_cache` is a generic wrapper that checks a cache before calling an arbitrary embedding function, used here for the sparse embedding path.
- `from agentic_rag.embedding.sparse_client import embed_sparse_texts` — imports the function that computes sparse embeddings (a keyword-weighted representation) for a batch of texts.
- `from agentic_rag.indexing.qdrant_setup import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME` — imports the string constants naming the dense and sparse vector fields as they're configured inside the Qdrant collection. Using shared constants (rather than hardcoding the strings here) keeps this file in sync with however the collection was actually set up during indexing.
- `from agentic_rag.retrieval.access import allowed_tiers_for` — imports the access-control helper from `access.py` that resolves which content tiers a given user is allowed to see.

### Lines 19-25 — The over-fetch factor constant
```python
# Qdrant's RRF fusion only ranks over what each leg's prefetch already
# returned. If prefetch limit == the final limit, a candidate ranked just
# outside top_k on BOTH legs individually - but competitive after fusion -
# would never be fetched at all. Over-fetching per leg is the standard fix;
# 4x is a common, conservative starting point with no established tuning
# need yet.
_PREFETCH_OVERFETCH_FACTOR = 4
```
- The comment explains a subtle but important correctness issue with RRF fusion: fusion can only combine and re-rank candidates that were actually retrieved by each individual search leg. If each leg (dense, sparse) only fetched exactly `top_k` candidates, then a chunk that individually ranked just outside the top `top_k` on *both* legs — but would have scored well once the two rankings are combined — would never even be considered, because it was never fetched in the first place. The standard fix is to "over-fetch": ask each leg for more candidates than you ultimately need, so fusion has a wider pool to work with.
- `_PREFETCH_OVERFETCH_FACTOR = 4` — defines that over-fetch multiplier as a module-level constant set to 4x, described as "a common, conservative starting point" rather than a value that's been carefully tuned for this specific system yet. Keeping it as a named constant (rather than inlining `top_k * 4` later) makes the value easy to find, adjust, and reason about independently of the search logic.

### Lines 28-34 — `SearchCandidate` data structure
```python
@dataclass(frozen=True)
class SearchCandidate:
    relative_path: str
    chunk_index: int
    text: str
    access_tier: str
    score: float
```
- `@dataclass(frozen=True)` — marks `SearchCandidate` as an immutable dataclass: once created, its fields can't be reassigned. This is a deliberate safety choice — search results shouldn't be silently mutated as they flow through reranking and generation; if a field like `score` needs to be updated (as `rerank.py` does), the correct pattern is to create a new copy via `dataclasses.replace` rather than mutate in place.
- `class SearchCandidate:` — represents one retrieved chunk of content, along with enough metadata to trace it back to its source and judge its relevance and permissions.
- `relative_path: str` — the file path (relative to some root) that this chunk of text came from, so results can be attributed back to a specific document.
- `chunk_index: int` — identifies which chunk, in order, this is within that source file (since a document is typically split into multiple chunks during indexing).
- `text: str` — the actual text content of this chunk, which is what gets shown to the user or fed into the generation step.
- `access_tier: str` — the access tier this chunk was tagged with at indexing time, used to confirm/report which permission level it belongs to.
- `score: float` — the chunk's relevance score, initially the fused hybrid search score from Qdrant, but potentially replaced later by a more accurate cross-encoder score during reranking (see `rerank.py`).

### Lines 37-50 — `hybrid_search` function signature
```python
def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    *,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout_seconds: int,
    sparse_model: str,
    embedding_cache: EmbeddingCache,
    user_tier: str,
    known_tiers: list[str],
    top_k: int,
) -> list[SearchCandidate]:
```
- `client: QdrantClient` — the already-configured Qdrant client to issue the search request through; passed in rather than constructed here, keeping this function focused purely on search logic and easy to test with a different client if needed.
- `collection_name: str` — which Qdrant collection to search within.
- `query: str` — the raw user query text to search for.
- `*,` — this marks everything after it as keyword-only arguments, meaning callers must write e.g. `embedding_model=...` explicitly rather than passing values positionally. With this many parameters of similar types (several strings), this prevents accidental mix-ups (like passing the sparse model where the embedding model was meant) and makes call sites self-documenting.
- `embedding_model: str` — which model name to use for computing the dense (semantic) query embedding via Ollama.
- `ollama_base_url: str` — the base URL of the Ollama server used to compute the dense embedding.
- `embedding_timeout_seconds: int` — how long to wait for the dense embedding request before timing out.
- `sparse_model: str` — which model to use to compute the sparse (keyword-style) embedding.
- `embedding_cache: EmbeddingCache` — a cache instance passed in so that repeated queries (or repeated calls with cache-worthy inputs) don't redundantly recompute embeddings.
- `user_tier: str` — the requesting user's access tier, used to determine which content they're allowed to see.
- `known_tiers: list[str]` — the full, ordered list of valid access tiers configured for the system, needed by `allowed_tiers_for` to resolve `user_tier` into the set of visible tiers.
- `top_k: int` — how many final results the caller wants back after fusion.
- `-> list[SearchCandidate]:` — the function returns a plain Python list of `SearchCandidate` objects, ready for the caller (e.g. the reranking step) to consume.

### Lines 51-62 — Docstring: what this function guarantees
```python
    """Dense + sparse search against Qdrant, fused natively (RRF) into a
    single ranked list of up to `top_k` candidates.

    Access filtering is applied to *both* the dense and sparse legs before
    fusion, not to the fused result afterward - a chunk the user isn't
    permitted to see must never influence the fused ranking or be
    returned, per REQUIREMENTS.md §11/FR3.

    Raises UnknownAccessTierError (via allowed_tiers_for) if `user_tier`
    isn't in `known_tiers` - a bad access tier must fail loudly, not
    silently search with no results or, worse, no filter at all.
    """
```
- The first paragraph summarizes the overall approach: run both dense and sparse search, and let Qdrant's built-in RRF fusion (rather than custom application code) merge the two rankings into one list capped at `top_k`.
- The second paragraph documents a critical design decision with its reasoning: access-tier filtering is applied to each search leg (dense and sparse) *before* fusion happens, not applied afterward by filtering the final fused list. This matters because if filtering were done only after fusion, a restricted chunk could still influence the fused ranking (e.g. by "crowding out" other legitimate results, or by affecting relative rank positions) even if it were stripped out at the very end — and worse, doing it after fusion risks a chunk with restricted content slipping into the returned list if the post-filter step were ever buggy or skipped. Filtering at the database query level, before fusion, guarantees restricted chunks are never fetched, never influence ranking, and never returned. This is explicitly tied back to a requirement (`REQUIREMENTS.md §11/FR3`) rather than being an arbitrary implementation choice.
- The third paragraph documents the error behavior: if `user_tier` isn't a recognized tier, the function (through its call to `allowed_tiers_for`) raises `UnknownAccessTierError` rather than degrading gracefully. The reasoning given mirrors `access.py`'s philosophy — silently returning no results, or worse, silently applying no filter at all (which could expose everything), are both worse failure modes than an explicit, loud error.

### Lines 63-66 — Resolving allowed tiers and building the access filter
```python
    allowed_tiers = allowed_tiers_for(user_tier, known_tiers)
    access_filter = Filter(
        must=[FieldCondition(key="access_tier", match=MatchAny(any=allowed_tiers))]
    )
```
- `allowed_tiers = allowed_tiers_for(user_tier, known_tiers)` — calls into `access.py` to resolve the full list of tiers this user is allowed to see (their own tier plus everything below it), given the configured tier hierarchy. This is also where `UnknownAccessTierError` would be raised and propagate up if `user_tier` were invalid.
- `access_filter = Filter(must=[FieldCondition(key="access_tier", match=MatchAny(any=allowed_tiers))])` — builds a Qdrant filter object that will be applied to search queries. `FieldCondition(key="access_tier", ...)` says "look at the `access_tier` field stored on each point's payload," and `MatchAny(any=allowed_tiers)` says "match if that field's value is any one of the tiers in `allowed_tiers`." Wrapping it in `Filter(must=[...])` makes this condition mandatory — only points whose `access_tier` is in the allowed list will be considered a match at all. This single filter object is reused for both the dense and sparse search legs below, ensuring identical access enforcement on each.

### Lines 68-75 — `embed_dense`: computing the dense query embedding
```python
    def embed_dense() -> list[float]:
        return embed_query_dense(
            query,
            model=embedding_model,
            base_url=ollama_base_url,
            timeout=embedding_timeout_seconds,
            cache=embedding_cache,
        )
```
- `def embed_dense() -> list[float]:` — defines a small local (nested) function with no arguments, capturing `query`, `embedding_model`, `ollama_base_url`, `embedding_timeout_seconds`, and `embedding_cache` from the enclosing scope. Wrapping this call in a zero-argument function makes it easy to hand off to a thread pool below (thread pool executors call functions with no arguments most simply this way).
- `return embed_query_dense(query, model=embedding_model, base_url=ollama_base_url, timeout=embedding_timeout_seconds, cache=embedding_cache)` — calls the imported dense-embedding helper, passing the query text and all the configuration needed to reach Ollama (which model to use, the server URL, how long to wait) plus the shared cache so repeated identical queries can be served without another network round-trip.

### Lines 77-83 — `embed_sparse`: computing the sparse query embedding
```python
    def embed_sparse():
        return embed_with_cache(
            [query],
            model=sparse_model,
            cache=embedding_cache,
            embed_fn=lambda batch: embed_sparse_texts(batch, model_name=sparse_model),
        )[0]
```
- `def embed_sparse():` — another zero-argument nested function, this time computing the sparse embedding for the query, for the same reason (easy to hand to the thread pool).
- `return embed_with_cache([query], model=sparse_model, cache=embedding_cache, embed_fn=lambda batch: embed_sparse_texts(batch, model_name=sparse_model), )[0]` — calls the generic caching wrapper `embed_with_cache`, passing the query wrapped in a single-element list (`[query]`), because the underlying sparse embedding machinery is designed to operate on batches of texts rather than one at a time. `embed_fn` is a small inline function (`lambda batch: embed_sparse_texts(batch, model_name=sparse_model)`) that tells the cache wrapper how to actually compute embeddings for whatever items in the batch aren't already cached, by delegating to `embed_sparse_texts`. Since the input was a one-item list, the result is also a one-item list, so `[0]` pulls out that single sparse embedding for the query.

### Lines 85-92 — Running dense and sparse embedding concurrently
```python
    # Dense embedding is a blocking Ollama HTTP round-trip; sparse is local
    # CPU work. Run them concurrently rather than paying both latencies
    # back-to-back on every query - this is the hottest path in the system.
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(embed_dense)
        sparse_future = executor.submit(embed_sparse)
        dense_vector = dense_future.result()
        sparse_vector = sparse_future.result()
```
- The comment explains why concurrency is used here specifically: the dense embedding call is a "blocking" network request to Ollama (meaning the program has to wait, doing nothing, until the HTTP response comes back), whereas the sparse embedding is local CPU work (no network wait). Since neither depends on the other's result, running them one after another would mean paying both delays in sequence on every single search, even though there's no reason they can't happen at the same time. The comment also flags this as "the hottest path in the system" — i.e., this code runs on essentially every user query, so shaving latency here has an outsized impact on overall responsiveness.
- `with ThreadPoolExecutor(max_workers=2) as executor:` — creates a thread pool limited to 2 worker threads (exactly enough for the two embedding tasks) and ensures it's properly shut down afterward via the `with` block (a context manager pattern).
- `dense_future = executor.submit(embed_dense)` — schedules the dense embedding function to run on a worker thread and immediately returns a "future" object representing that in-progress (or eventually completed) computation, without blocking the main thread yet.
- `sparse_future = executor.submit(embed_sparse)` — similarly schedules the sparse embedding function to start running, essentially at the same time as the dense one, on the pool's second thread.
- `dense_vector = dense_future.result()` — blocks (waits) until the dense embedding task finishes, then retrieves its return value.
- `sparse_vector = sparse_future.result()` — likewise waits for and retrieves the sparse embedding result. Because both tasks were already submitted and running concurrently before either `.result()` call, the total wait time is roughly the *longer* of the two tasks' durations, not the sum of both.

### Line 94 — Computing the prefetch limit
```python
    prefetch_limit = top_k * _PREFETCH_OVERFETCH_FACTOR
```
- `prefetch_limit = top_k * _PREFETCH_OVERFETCH_FACTOR` — multiplies the caller's requested `top_k` by the over-fetch constant defined near the top of the file (4), producing the number of candidates each individual search leg (dense, sparse) should retrieve *before* fusion narrows things back down to `top_k`. This directly implements the over-fetching strategy explained in that constant's comment.

### Lines 96-115 — Issuing the hybrid search query to Qdrant
```python
    result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=access_filter,
            ),
            Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=access_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
```
- `result = client.query_points(collection_name=collection_name, ...)` — issues a single, multi-stage query to the Qdrant collection, using Qdrant's native "query with prefetch + fusion" capability rather than running two separate searches and merging them in application code. Doing the fusion natively in Qdrant is both simpler and lets the database engine optimize the whole operation.
- `prefetch=[Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=prefetch_limit, filter=access_filter), Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=prefetch_limit, filter=access_filter), ]` — defines the two "legs" of the search that will run before fusion. The first `Prefetch` searches using the `dense_vector` computed earlier against the named dense vector field (`DENSE_VECTOR_NAME`, imported from `qdrant_setup.py` so it always matches how the collection was actually built), fetching up to `prefetch_limit` candidates, restricted by `access_filter`. The second `Prefetch` does the equivalent for the sparse vector, using `sparse_vector` and `SPARSE_VECTOR_NAME`, also with the same `prefetch_limit` and the same `access_filter`. Applying `access_filter` identically to both legs is what makes the access-control guarantee described in the docstring hold — no restricted content can enter either leg's candidate pool in the first place.
- `query=FusionQuery(fusion=Fusion.RRF),` — tells Qdrant that after running the two prefetch legs, it should combine their results using the RRF (Reciprocal Rank Fusion) algorithm, which produces a single ranking by combining each candidate's rank position across the two individual result lists (favoring items that rank well in either or both), rather than trying to directly compare dense similarity scores to sparse similarity scores (which aren't on the same numeric scale and so can't be compared or averaged meaningfully).
- `limit=top_k,` — caps the final, fused result list at the caller's requested `top_k`, distinct from the larger `prefetch_limit` used for each individual leg.
- `with_payload=True,` — instructs Qdrant to include each matching point's stored payload data (the metadata like `relative_path`, `chunk_index`, `text`, `access_tier`) in the response, since the code needs that information to build `SearchCandidate` objects below; without this flag, Qdrant would return only IDs and scores.

### Lines 117-126 — Converting Qdrant results into `SearchCandidate` objects
```python
    return [
        SearchCandidate(
            relative_path=point.payload["relative_path"],
            chunk_index=point.payload["chunk_index"],
            text=point.payload["text"],
            access_tier=point.payload["access_tier"],
            score=point.score,
        )
        for point in result.points
    ]
```
- `return [SearchCandidate(...) for point in result.points]` — a list comprehension that iterates over every point (matching result) Qdrant returned in `result.points`, and converts each one from Qdrant's generic point representation into this codebase's own well-typed `SearchCandidate` dataclass, so downstream code (reranking, generation) doesn't need to know anything about Qdrant's internal response format.
- `relative_path=point.payload["relative_path"]` — pulls the source file path out of the point's payload dictionary.
- `chunk_index=point.payload["chunk_index"]` — pulls the chunk's position within its source document out of the payload.
- `text=point.payload["text"]` — pulls the actual chunk text out of the payload.
- `access_tier=point.payload["access_tier"]` — pulls the tier this chunk was tagged with out of the payload (useful for traceability/auditing, even though the filtering that determined the chunk was allowed already happened earlier in the query).
- `score=point.score` — takes the fused RRF score Qdrant computed for this point and stores it as the candidate's initial relevance score (which, as noted in `rerank.py`, may later be overwritten with a more accurate cross-encoder score during reranking).
