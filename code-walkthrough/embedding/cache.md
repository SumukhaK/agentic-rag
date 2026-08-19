# `embedding/cache.py`

**Purpose:** This file provides an in-memory cache so the same piece of text is never sent to the embedding model twice within a single run of the program. Turning text into an "embedding" (a list of numbers that represents its meaning so it can be compared to other texts mathematically) is relatively slow and costly, since it means making a network call to Ollama (a locally-running LLM server) for every text. If the same text shows up more than once — for example, the same query embedded for both a dense and a semantic-cache lookup — this file lets the second request reuse the first result instead of recomputing it. It also provides a small helper function specifically for embedding a single search query through that cache, since that exact pattern is needed in more than one place in the system.

## Line-by-line walkthrough

### Lines 1-5 — Imports and generic type setup
```python
from typing import Callable, TypeVar

from agentic_rag.embedding.ollama_client import embed_texts

T = TypeVar("T")
```
- `from typing import Callable, TypeVar` — imports two typing helpers: `Callable`, used later to describe "a function that takes a list of texts and returns a list of something," and `TypeVar`, used to create a placeholder type that can stand for different concrete types depending on how it's used.
- `from agentic_rag.embedding.ollama_client import embed_texts` — imports the function that actually talks to Ollama's embedding endpoint (defined in `ollama_client.py`), so this file can use it as the default way of computing embeddings when nothing is cached yet.
- `T = TypeVar("T")` — defines a generic type variable `T`. This is used so the same caching function can work whether it's caching `list[float]` (a "dense" embedding, i.e. one long vector of numbers) or Qdrant's `SparseVector` type (a "sparse" embedding, i.e. mostly-zero vectors represented by just their non-zero positions and values) — the type checker will infer `T` from whatever function is passed in.

### Lines 8-19 — `EmbeddingCache` class definition and constructor
```python
class EmbeddingCache:
    """In-memory cache from (model, text) to an already-computed embedding.

    Scoped to the process's lifetime, not persisted across restarts - see
    docs/REQUIREMENTS.md §7 for why that's a deliberate, documented choice
    for this first version rather than an oversight. Also unbounded (no
    eviction) - fine for one sync cycle at a time, but §7 already flags
    eviction policy as an explicit open item, not something to invent here.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], object] = {}
```
- `class EmbeddingCache:` — defines a class whose job is to remember embeddings that have already been computed, so they don't need to be recomputed.
- The docstring explains two deliberate design limits: the cache lives only as long as the running process (it's not saved to disk, so it starts empty every time the program restarts), and it never removes old entries ("no eviction"). Both of these are documented as known, intentional trade-offs for the current version of the project rather than bugs — good enough for a single data-sync run, with a note that a smarter eviction policy is left as future work.
- `def __init__(self) -> None:` — the constructor, run once when an `EmbeddingCache` object is created.
- `self._store: dict[tuple[str, str], object] = {}` — sets up the actual storage: a dictionary whose keys are `(model, text)` pairs (a tuple, i.e. a fixed pair of values) and whose values are the computed embedding (typed loosely as `object` since it could be either a dense vector or a sparse vector). Keying by both the model name and the text is important — the same text embedded by two different models would produce two different, non-interchangeable vectors, so the model name has to be part of the key or the cache could return the wrong kind of result.

### Lines 21-25 — `get` and `set` methods
```python
    def get(self, model: str, text: str) -> object | None:
        return self._store.get((model, text))

    def set(self, model: str, text: str, value: object) -> None:
        self._store[(model, text)] = value
```
- `def get(self, model: str, text: str) -> object | None:` — looks up a previously cached embedding for a given `(model, text)` pair.
- `return self._store.get((model, text))` — uses the dictionary's `.get()` method, which returns `None` if the key isn't present instead of raising an error — this makes it easy for callers to check "was this cached?" without a try/except block.
- `def set(self, model: str, text: str, value: object) -> None:` — stores a newly computed embedding under its `(model, text)` key.
- `self._store[(model, text)] = value` — writes (or overwrites) the entry in the dictionary.

### Lines 28-33 — `embed_with_cache` signature
```python
def embed_with_cache(
    texts: list[str],
    model: str,
    cache: EmbeddingCache,
    embed_fn: Callable[[list[str]], list[T]],
) -> list[T]:
```
- This defines the core caching function. It takes: `texts` (the list of strings to embed), `model` (which embedding model to use — part of the cache key), `cache` (the `EmbeddingCache` instance to read from and write to), and `embed_fn` — a function to call for whatever texts aren't already cached, which takes a list of texts and returns a list of embeddings of type `T`. Because `embed_fn` and the return type are both written in terms of `T`, the function works generically for dense embeddings, sparse embeddings, or any other future embedding type, as long as the caller supplies a matching `embed_fn`.

### Lines 34-41 — Docstring explaining the generic design
```python
    """Embed `texts`, reusing cached results and only calling `embed_fn`
    for the texts not already cached under (model, text).

    Generic over what `embed_fn` returns, so the same cache and wrapper
    work for both dense (list[float]) and sparse (SparseVector) embedding
    clients - the cache key already discriminates by model, so a single
    shared EmbeddingCache instance is safe to use for both.
    """
```
- Explains the function's behavior in plain terms: reuse what's cached, only compute what's missing. It also explains why one `EmbeddingCache` instance can safely be shared between dense and sparse embedding call sites — because the key already includes the model name, a dense model's entries and a sparse model's entries can never collide or be confused with each other, even though they live in the same dictionary.

### Lines 42-43 — Early exit for empty input
```python
    if not texts:
        return []
```
- If there's nothing to embed, return an empty list immediately rather than doing any lookup or calling `embed_fn` with an empty list — a small guard against unnecessary work (and against `embed_fn` implementations that might not handle an empty list gracefully).

### Lines 45-46 — Checking the cache for every requested text
```python
    results: list[T | None] = [cache.get(model, text) for text in texts]  # type: ignore[misc]
    missing_indices = [i for i, result in enumerate(results) if result is None]
```
- `results: list[T | None] = [cache.get(model, text) for text in texts]` — builds a list the same length and order as `texts`, where each entry is either the cached embedding for that text (if it was already computed before) or `None` (if it hasn't been seen yet). The `# type: ignore[misc]` comment tells the type checker to not complain here — `cache.get` is declared to return `object | None`, not `T | None`, so this line is technically imprecise from the type checker's point of view even though it's correct in practice.
- `missing_indices = [i for i, result in enumerate(results) if result is None]` — records the positions (indices) in the list where nothing was found in the cache, i.e. the texts that still need to be embedded.

### Lines 48-64 — Embedding the missing texts and merging results
```python
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        fresh = embed_fn(missing_texts)

        if len(fresh) != len(missing_texts):
            # Validate before caching anything from this batch: a partial
            # result cached now would look like a legitimate hit on retry,
            # masking the same malformed-response class of bug this check
            # exists to catch.
            raise ValueError(
                f"embed_fn returned {len(fresh)} result(s) for "
                f"{len(missing_texts)} requested text(s)"
            )

        for index, vector in zip(missing_indices, fresh):
            cache.set(model, texts[index], vector)
            results[index] = vector
```
- `if missing_indices:` — only does the extra work below if there's actually at least one text that wasn't already cached; if everything was a cache hit, this whole block is skipped.
- `missing_texts = [texts[i] for i in missing_indices]` — pulls out just the actual text strings that need embedding, using the indices gathered above.
- `fresh = embed_fn(missing_texts)` — calls the supplied embedding function (e.g. `embed_texts` for dense vectors, or a sparse equivalent) to compute embeddings only for the texts that were missing — this is the whole point of the cache, avoiding recomputation of texts already seen.
- `if len(fresh) != len(missing_texts):` — a safety check: the code assumes `embed_fn` returns exactly one result per text it was given, in the same order. If it doesn't (e.g. the underlying API silently dropped or merged some entries), this check catches that instead of silently mismatching results to the wrong texts.
- The comment explains *why* this check happens before anything is written to the cache: if a mismatched (partial) batch were cached anyway, a later retry could see those bad entries as "already cached" and would never notice the earlier problem — so validating first prevents the bug from being hidden.
- `raise ValueError(...)` — if the lengths don't match, raise an error with a clear message stating exactly how many results came back versus how many were requested, making the mismatch easy to diagnose.
- `for index, vector in zip(missing_indices, fresh):` — walks through the newly computed embeddings alongside the positions they belong to (`zip` pairs them up in order, relying on the same-length, same-order assumption just validated above).
- `cache.set(model, texts[index], vector)` — stores each freshly computed embedding in the cache under its `(model, text)` key, so future calls for the same text won't need to recompute it.
- `results[index] = vector` — fills in the previously-`None` slot in the `results` list with the newly computed vector, so the final list returned to the caller is complete and in the original order.

### Line 66 — Returning the combined results
```python
    return results  # type: ignore[return-value]
```
- Returns the `results` list, which by this point is fully populated — a mix of values that came from the cache and values that were just computed. The `# type: ignore[return-value]` comment silences the type checker because `results` is typed as `list[T | None]` but by now every `None` has been replaced, so the actual returned value truthfully matches the declared `list[T]` return type even though the checker can't prove it.

### Lines 69-71 — `embed_query_dense` signature
```python
def embed_query_dense(
    query: str, *, model: str, base_url: str, timeout: int, cache: EmbeddingCache
) -> list[float]:
```
- Defines a convenience function specifically for embedding one query string as a dense vector (a `list[float]`). The `*` in the parameter list forces every argument after it (`model`, `base_url`, `timeout`, `cache`) to be passed by name (as `model=...`, etc.) rather than by position, which avoids mistakes from accidentally swapping same-typed arguments like `base_url` and a stray string.

### Lines 72-76 — Docstring explaining why this exists
```python
    """Embed a single query string as a dense vector, through the shared
    cache - the common "one query, cache-aware" pattern used by both
    hybrid search's dense leg and the semantic cache's lookup embedding.
    Pulled out once both needed the identical embed_with_cache+embed_texts
    wiring, rather than kept duplicated across two call sites."""
```
- Explains the reason this small helper function exists at all: two different parts of the system ("hybrid search," which combines dense and sparse search, and a "semantic cache" that looks up similar past queries) both need to turn a single query string into a cached dense embedding in exactly the same way. Rather than writing that same few lines of wiring twice, it was factored out into one shared function.

### Lines 77-82 — Function body
```python
    return embed_with_cache(
        [query],
        model=model,
        cache=cache,
        embed_fn=lambda batch: embed_texts(batch, model=model, base_url=base_url, timeout=timeout),
    )[0]
```
- `return embed_with_cache(...)` — delegates to the generic caching function defined above rather than duplicating any caching logic.
- `[query]` — wraps the single query string in a one-element list, because `embed_with_cache` (and `embed_fn`) are built around lists of texts, not single strings.
- `model=model, cache=cache` — passes through the model name and shared cache instance so this query's embedding is looked up and stored using the same cache as everything else.
- `embed_fn=lambda batch: embed_texts(batch, model=model, base_url=base_url, timeout=timeout)` — supplies the actual "how to compute an embedding if it's not cached" function. It's a small anonymous function (`lambda`) that, when called with a list of texts (`batch`), calls the real Ollama-backed `embed_texts` function imported at the top of the file, passing along the model name, the server's base URL, and the timeout (how many seconds to wait before giving up on the request).
- `[0]` — `embed_with_cache` always returns a list (one entry per input text), but since this function only ever passes in a single-element list (`[query]`), it unwraps that list and returns just the one embedding directly, matching the declared `list[float]` return type.
