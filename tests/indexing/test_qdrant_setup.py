from qdrant_client.models import Distance

from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client


def test_ensure_collection_creates_a_new_collection_with_the_given_vector_size(
    tmp_path,
):
    client = get_client(str(tmp_path))

    ensure_collection(client, collection_name="documents", vector_size=768)

    assert client.collection_exists("documents")
    info = client.get_collection("documents")
    assert info.config.params.vectors.size == 768
    assert info.config.params.vectors.distance == Distance.COSINE


def test_ensure_collection_is_idempotent_when_already_exists(tmp_path):
    client = get_client(str(tmp_path))

    ensure_collection(client, collection_name="documents", vector_size=768)
    ensure_collection(client, collection_name="documents", vector_size=768)  # no raise

    assert client.collection_exists("documents")


def test_get_client_uses_local_embedded_storage_at_the_given_path(tmp_path):
    client = get_client(str(tmp_path))

    ensure_collection(client, collection_name="documents", vector_size=768)

    # Local/embedded mode persists to disk at the given path - no server
    # process, no network call.
    assert any(tmp_path.iterdir())
