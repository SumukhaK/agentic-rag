from agentic_rag.embedding.cache import EmbeddingCache, embed_with_cache


def test_embed_with_cache_calls_embed_fn_only_for_uncached_texts():
    cache = EmbeddingCache()
    calls = []

    def embed_fn(batch):
        calls.append(list(batch))
        return [[len(text)] for text in batch]

    embed_with_cache(["one", "two"], model="m", cache=cache, embed_fn=embed_fn)
    embed_with_cache(["two", "three"], model="m", cache=cache, embed_fn=embed_fn)

    assert calls == [["one", "two"], ["three"]]


def test_embed_with_cache_returns_correct_order_for_mixed_cached_and_uncached():
    cache = EmbeddingCache()
    embed_fn = lambda batch: [[len(text)] for text in batch]  # noqa: E731

    embed_with_cache(["one", "two"], model="m", cache=cache, embed_fn=embed_fn)
    result = embed_with_cache(
        ["zero", "one", "two"], model="m", cache=cache, embed_fn=embed_fn
    )

    assert result == [[4], [3], [3]]


def test_embed_with_cache_stores_results_for_reuse():
    cache = EmbeddingCache()
    embed_fn = lambda batch: [[42] for _ in batch]  # noqa: E731

    embed_with_cache(["one"], model="m", cache=cache, embed_fn=embed_fn)

    assert cache.get("m", "one") == [42]


def test_embed_with_cache_returns_empty_list_for_no_input():
    cache = EmbeddingCache()

    def embed_fn(batch):
        raise AssertionError("embed_fn should not be called for empty input")

    assert embed_with_cache([], model="m", cache=cache, embed_fn=embed_fn) == []


def test_embed_with_cache_keys_are_scoped_per_model():
    cache = EmbeddingCache()
    calls = []

    def embed_fn(batch):
        calls.append(list(batch))
        return [[1] for _ in batch]

    embed_with_cache(["text"], model="dense-model", cache=cache, embed_fn=embed_fn)
    embed_with_cache(["text"], model="sparse-model", cache=cache, embed_fn=embed_fn)

    # Same text, different model - both must be embedded, not conflated.
    assert calls == [["text"], ["text"]]
