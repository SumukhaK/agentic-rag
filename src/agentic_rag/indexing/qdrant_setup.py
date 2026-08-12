from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def get_client(storage_path: str) -> QdrantClient:
    """Local/embedded Qdrant client: on-disk storage at `storage_path`, no
    server process. Docker isn't available in this dev environment - see
    docs/REQUIREMENTS.md §5. Swappable for a real server later by passing
    a `url=` instead of `path=` here.
    """
    return QdrantClient(path=storage_path)


def ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    """Create `collection_name` if it doesn't already exist. Idempotent.

    Qdrant indexes dense vectors with HNSW by default - there's no
    alternative index to opt into, so creating the collection with its
    default vector config already satisfies the HNSW requirement.
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
