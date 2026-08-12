import pytest

from agentic_rag.embedding.sparse_client import (
    SparseEmbeddingError,
    embed_sparse_texts,
)

MODEL = "Qdrant/bm25"


@pytest.fixture(scope="module", autouse=True)
def _require_sparse_model():
    """fastembed downloads the BM25 tokenizer/vocab bundle to the OS temp
    directory on first use, not a stable cache location - so on a fresh
    environment with no network, that download fails. Skip these tests
    with a clear reason rather than failing confusingly, since this is an
    environment precondition, not a bug in the code under test."""
    try:
        embed_sparse_texts(["warmup"], model_name=MODEL)
    except SparseEmbeddingError as exc:
        pytest.skip(f"sparse embedding model unavailable: {exc}")


def test_embed_sparse_texts_returns_one_vector_per_input():
    result = embed_sparse_texts(
        ["Arsenal drew 1-1.", "Chelsea won 3-0."], model_name=MODEL
    )

    assert len(result) == 2
    for vector in result:
        assert len(vector.indices) > 0
        assert len(vector.indices) == len(vector.values)


def test_embed_sparse_texts_gives_different_vectors_for_different_texts():
    result = embed_sparse_texts(
        ["Arsenal drew 1-1.", "Completely unrelated sentence about weather."],
        model_name=MODEL,
    )

    assert result[0].indices != result[1].indices


def test_embed_sparse_texts_is_deterministic_regardless_of_batch_composition():
    alone = embed_sparse_texts(["Arsenal drew 1-1."], model_name=MODEL)[0]
    in_batch = embed_sparse_texts(
        ["Arsenal drew 1-1.", "some other text"], model_name=MODEL
    )[0]

    assert alone.indices == in_batch.indices
    assert alone.values == in_batch.values


def test_embed_sparse_texts_returns_empty_list_for_no_input():
    assert embed_sparse_texts([], model_name=MODEL) == []


def test_embed_sparse_texts_raises_sparse_embedding_error_for_an_unknown_model():
    with pytest.raises(SparseEmbeddingError):
        embed_sparse_texts(["Arsenal drew 1-1."], model_name="not-a-real-model")
