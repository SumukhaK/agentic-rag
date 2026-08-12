import pytest
from qdrant_client.models import Distance

from agentic_rag.indexing.qdrant_setup import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    CollectionSchemaMismatchError,
    ensure_collection,
    get_client,
)


def test_ensure_collection_creates_named_dense_and_sparse_vectors(tmp_path):
    client = get_client(tmp_path)

    ensure_collection(client, collection_name="documents", vector_size=768)

    info = client.get_collection("documents")
    dense = info.config.params.vectors[DENSE_VECTOR_NAME]
    assert dense.size == 768
    assert dense.distance == Distance.COSINE
    assert SPARSE_VECTOR_NAME in info.config.params.sparse_vectors


def test_ensure_collection_is_idempotent_when_already_exists(tmp_path):
    client = get_client(tmp_path)

    ensure_collection(client, collection_name="documents", vector_size=768)
    ensure_collection(client, collection_name="documents", vector_size=768)  # no raise

    assert client.collection_exists("documents")


def test_ensure_collection_raises_when_existing_vector_size_does_not_match(tmp_path):
    client = get_client(tmp_path)
    ensure_collection(client, collection_name="documents", vector_size=768)

    with pytest.raises(CollectionSchemaMismatchError):
        ensure_collection(client, collection_name="documents", vector_size=1024)


def test_get_client_uses_local_embedded_storage_at_the_given_path(tmp_path):
    client = get_client(tmp_path)

    ensure_collection(client, collection_name="documents", vector_size=768)

    # Local/embedded mode persists to disk: the storage path is non-empty
    # after creating a collection.
    assert any(tmp_path.iterdir())
