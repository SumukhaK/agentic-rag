import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from agentic_rag.embedding.ollama_client import embed_texts
from agentic_rag.embedding.sparse_client import embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from agentic_rag.ingestion.pipeline import IngestedDocument

# Fixed, arbitrary namespace for deterministic point IDs - only needs to be
# stable across runs of this codebase, not globally unique.
_POINT_ID_NAMESPACE = uuid.UUID("5c1e6f5a-6b8e-4b8a-9b0a-8f1d2c3e4f5a")


def _point_id(relative_path: str, chunk_index: int) -> str:
    """Deterministic per (document, chunk): re-indexing the same chunk
    always produces the same point ID, making index_document() safe to
    retry - running it twice for the same document converges to the same
    set of points instead of accumulating duplicates."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{relative_path}::{chunk_index}"))


def _path_filter(relative_path: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="relative_path", match=MatchValue(value=relative_path))]
    )


def delete_document(client: QdrantClient, collection_name: str, relative_path: str) -> None:
    """Remove every indexed point for `relative_path`. A no-op if none exist."""
    client.delete(collection_name=collection_name, points_selector=_path_filter(relative_path))


def index_document(
    client: QdrantClient,
    collection_name: str,
    document: IngestedDocument,
    *,
    embedding_model: str,
    ollama_base_url: str,
    sparse_model: str,
    embedding_timeout_seconds: int = 30,
) -> None:
    """Embed every chunk of `document` (dense + sparse) and upsert it as a
    Qdrant point, with the payload citation/access-control needs at query
    time.

    Existing points for this document's relative_path are deleted first,
    then the current chunk set is inserted fresh. This is what keeps an
    edit from leaving orphaned points behind: if a document shrinks from 5
    chunks to 3, only upserting the new 3 (without first clearing what was
    there) would leave chunks 3-4 stale and searchable forever.
    """
    delete_document(client, collection_name, document.relative_path)

    if not document.chunks:
        return

    texts = [chunk.text for chunk in document.chunks]
    dense_vectors = embed_texts(
        texts,
        model=embedding_model,
        base_url=ollama_base_url,
        timeout=embedding_timeout_seconds,
    )
    sparse_vectors = embed_sparse_texts(texts, model_name=sparse_model)

    points = [
        PointStruct(
            id=_point_id(document.relative_path, chunk.index),
            vector={DENSE_VECTOR_NAME: dense, SPARSE_VECTOR_NAME: sparse},
            payload={
                "relative_path": document.relative_path,
                "chunk_index": chunk.index,
                "text": chunk.text,
                "access_tier": document.access_tier,
            },
        )
        for chunk, dense, sparse in zip(document.chunks, dense_vectors, sparse_vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)
