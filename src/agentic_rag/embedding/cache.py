import hashlib
from typing import Callable, TypeVar

T = TypeVar("T")


class EmbeddingCache:
    """In-memory cache from (model, text) to an already-computed embedding.

    Scoped to the process's lifetime, not persisted across restarts - see
    docs/REQUIREMENTS.md §7 for why that's a deliberate, documented choice
    for this first version rather than an oversight.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], object] = {}

    def _key(self, model: str, text: str) -> tuple[str, str]:
        return (model, hashlib.sha256(text.encode()).hexdigest())

    def get(self, model: str, text: str) -> object | None:
        return self._store.get(self._key(model, text))

    def set(self, model: str, text: str, value: object) -> None:
        self._store[self._key(model, text)] = value


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
        for index, vector in zip(missing_indices, fresh):
            cache.set(model, texts[index], vector)
            results[index] = vector

    return results  # type: ignore[return-value]
