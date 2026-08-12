from unittest.mock import patch

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from agentic_rag.embedding.sparse_client import SparseEmbeddingError, embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.indexing.upsert import delete_document, index_document
from agentic_rag.ingestion.chunker import Chunk
from agentic_rag.ingestion.pipeline import IngestedDocument

SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "documents"


@pytest.fixture(scope="module", autouse=True)
def _require_sparse_model():
    try:
        embed_sparse_texts(["warmup"], model_name=SPARSE_MODEL)
    except SparseEmbeddingError as exc:
        pytest.skip(f"sparse embedding model unavailable: {exc}")


@pytest.fixture
def client(tmp_path):
    client = get_client(tmp_path)
    ensure_collection(client, collection_name=COLLECTION, vector_size=3)
    return client


def _document(relative_path="tier-1/a.txt", chunk_texts=("Arsenal drew 1-1.",)):
    return IngestedDocument(
        relative_path=relative_path,
        markdown="\n\n".join(chunk_texts),
        chunks=[Chunk(text=text, index=i) for i, text in enumerate(chunk_texts)],
        access_tier="tier-1",
    )


def _count_points_for(client, relative_path):
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="relative_path", match=MatchValue(value=relative_path))
            ]
        ),
        limit=100,
    )
    return len(points)


def _index(client, document, mock_embed_texts, dense_vectors):
    mock_embed_texts.return_value = dense_vectors
    index_document(
        client,
        collection_name=COLLECTION,
        document=document,
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
        sparse_model=SPARSE_MODEL,
    )


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_index_document_upserts_one_point_per_chunk(mock_embed_texts, client):
    document = _document(chunk_texts=("Arsenal drew 1-1.", "Chelsea won 3-0."))

    _index(client, document, mock_embed_texts, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    assert _count_points_for(client, "tier-1/a.txt") == 2


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_index_document_sets_dense_sparse_and_payload_on_each_point(
    mock_embed_texts, client
):
    document = _document()

    _index(client, document, mock_embed_texts, [[0.1, 0.2, 0.3]])

    points, _ = client.scroll(
        collection_name=COLLECTION, with_vectors=True, with_payload=True, limit=10
    )
    point = points[0]
    # Qdrant normalizes stored vectors for Cosine-distance collections, so
    # compare direction (proportions), not raw magnitude.
    dense = point.vector["dense"]
    assert len(dense) == 3
    assert dense[0] / dense[1] == pytest.approx(0.1 / 0.2)
    assert dense[1] / dense[2] == pytest.approx(0.2 / 0.3)
    assert len(point.vector["sparse"].indices) > 0
    assert point.payload["relative_path"] == "tier-1/a.txt"
    assert point.payload["chunk_index"] == 0
    assert point.payload["text"] == "Arsenal drew 1-1."
    assert point.payload["access_tier"] == "tier-1"


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_index_document_reindexing_with_fewer_chunks_removes_stale_points(
    mock_embed_texts, client
):
    original = _document(chunk_texts=("one", "two", "three"))
    _index(client, original, mock_embed_texts, [[0.1, 0.2, 0.3]] * 3)
    assert _count_points_for(client, "tier-1/a.txt") == 3

    edited = _document(chunk_texts=("one",))
    _index(client, edited, mock_embed_texts, [[0.1, 0.2, 0.3]])

    assert _count_points_for(client, "tier-1/a.txt") == 1


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_delete_document_removes_all_points_for_that_path_only(
    mock_embed_texts, client
):
    doc_a = _document(relative_path="tier-1/a.txt")
    doc_b = _document(relative_path="tier-1/b.txt")
    _index(client, doc_a, mock_embed_texts, [[0.1, 0.2, 0.3]])
    _index(client, doc_b, mock_embed_texts, [[0.1, 0.2, 0.3]])

    delete_document(client, collection_name=COLLECTION, relative_path="tier-1/a.txt")

    assert _count_points_for(client, "tier-1/a.txt") == 0
    assert _count_points_for(client, "tier-1/b.txt") == 1


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_delete_document_is_a_no_op_when_nothing_indexed_yet(mock_embed_texts, client):
    delete_document(client, collection_name=COLLECTION, relative_path="tier-1/a.txt")  # no raise
