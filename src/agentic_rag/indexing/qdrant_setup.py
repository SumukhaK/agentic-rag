from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class CollectionSchemaMismatchError(Exception):
    """Raised when an existing collection's dense vector size doesn't match
    what's being requested - e.g. the embedding model/dimensions changed
    without the collection being migrated."""


def get_client(storage_path: str | Path) -> QdrantClient:
    """Local/embedded Qdrant client: on-disk storage at `storage_path`, no
    server process. Docker isn't available in this dev environment - see
    docs/REQUIREMENTS.md §5. Swappable for a real server later by passing
    a `url=` instead of `path=` here.
    """
    return QdrantClient(path=str(storage_path))


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> None:
    """Create `collection_name` if it doesn't already exist. Idempotent.

    Created with a named dense vector ("dense") *and* a named sparse
    vector ("sparse") from the start, even though sparse vectors aren't
    populated until native hybrid search (the next Phase 2 item) is wired
    in - Qdrant can't add a sparse vector field to an existing collection
    after creation, only recreate it, so setting the schema up right here
    avoids a full reindex later.

    Qdrant indexes dense vectors with HNSW by default - there's no
    alternative index to opt into, so this already satisfies the HNSW
    requirement with no manual tuning needed.

    Raises CollectionSchemaMismatchError if the collection already exists
    with a different dense vector size than requested, rather than
    silently leaving the mismatched collection in place.
    """
    if client.collection_exists(collection_name):
        existing = client.get_collection(collection_name).config.params.vectors[
            DENSE_VECTOR_NAME
        ]
        if existing.size != vector_size:
            raise CollectionSchemaMismatchError(
                f"'{collection_name}' already exists with dense vector size "
                f"{existing.size}, but {vector_size} was requested"
            )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=distance)
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
    )
