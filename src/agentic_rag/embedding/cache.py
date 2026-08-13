from typing import Callable, TypeVar

from agentic_rag.embedding.ollama_client import embed_texts

T = TypeVar("T")


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

    def get(self, model: str, text: str) -> object | None:
        return self._store.get((model, text))

    def set(self, model: str, text: str, value: object) -> None:
        self._store[(model, text)] = value


def embed_with_cache(
    texts: list[str],
    model: str,
    cache: EmbeddingCache,
    embed_fn: Callable[[list[str]], list[T]],
) -> list[T]:
    """Embed `texts`, reusing cached results and only calling `embed_fn`
    for the texts not already cached under (model, text).

    Generic over what `embed_fn` returns, so the same cache and wrapper
    work for both dense (list[float]) and sparse (SparseVector) embedding
    clients - the cache key already discriminates by model, so a single
    shared EmbeddingCache instance is safe to use for both.
    """
    if not texts:
        return []

    results: list[T | None] = [cache.get(model, text) for text in texts]  # type: ignore[misc]
    missing_indices = [i for i, result in enumerate(results) if result is None]

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

    return results  # type: ignore[return-value]


def embed_query_dense(
    query: str, *, model: str, base_url: str, timeout: int, cache: EmbeddingCache
) -> list[float]:
    """Embed a single query string as a dense vector, through the shared
    cache - the common "one query, cache-aware" pattern used by both
    hybrid search's dense leg and the semantic cache's lookup embedding.
    Pulled out once both needed the identical embed_with_cache+embed_texts
    wiring, rather than kept duplicated across two call sites."""
    return embed_with_cache(
        [query],
        model=model,
        cache=cache,
        embed_fn=lambda batch: embed_texts(batch, model=model, base_url=base_url, timeout=timeout),
    )[0]
