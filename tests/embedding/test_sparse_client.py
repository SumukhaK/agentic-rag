from agentic_rag.embedding.sparse_client import embed_sparse_texts

MODEL = "Qdrant/bm25"


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
