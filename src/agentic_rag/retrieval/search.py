from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    Prefetch,
)

from agentic_rag.embedding.cache import EmbeddingCache, embed_with_cache
from agentic_rag.embedding.ollama_client import embed_texts
from agentic_rag.embedding.sparse_client import embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from agentic_rag.retrieval.access import allowed_tiers_for

# Qdrant's RRF fusion only ranks over what each leg's prefetch already
# returned. If prefetch limit == the final limit, a candidate ranked just
# outside top_k on BOTH legs individually - but competitive after fusion -
# would never be fetched at all. Over-fetching per leg is the standard fix;
# 4x is a common, conservative starting point with no established tuning
# need yet.
_PREFETCH_OVERFETCH_FACTOR = 4


@dataclass(frozen=True)
class SearchCandidate:
    relative_path: str
    chunk_index: int
    text: str
    access_tier: str
    score: float


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    *,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout_seconds: int,
    sparse_model: str,
    embedding_cache: EmbeddingCache,
    user_tier: str,
    known_tiers: list[str],
    top_k: int,
) -> list[SearchCandidate]:
    """Dense + sparse search against Qdrant, fused natively (RRF) into a
    single ranked list of up to `top_k` candidates.

    Access filtering is applied to *both* the dense and sparse legs before
    fusion, not to the fused result afterward - a chunk the user isn't
    permitted to see must never influence the fused ranking or be
    returned, per REQUIREMENTS.md §11/FR3.

    Raises UnknownAccessTierError (via allowed_tiers_for) if `user_tier`
    isn't in `known_tiers` - a bad access tier must fail loudly, not
    silently search with no results or, worse, no filter at all.
    """
    allowed_tiers = allowed_tiers_for(user_tier, known_tiers)
    access_filter = Filter(
        must=[FieldCondition(key="access_tier", match=MatchAny(any=allowed_tiers))]
    )

    def embed_dense() -> list[float]:
        return embed_with_cache(
            [query],
            model=embedding_model,
            cache=embedding_cache,
            embed_fn=lambda batch: embed_texts(
                batch,
                model=embedding_model,
                base_url=ollama_base_url,
                timeout=embedding_timeout_seconds,
            ),
        )[0]

    def embed_sparse():
        return embed_with_cache(
            [query],
            model=sparse_model,
            cache=embedding_cache,
            embed_fn=lambda batch: embed_sparse_texts(batch, model_name=sparse_model),
        )[0]

    # Dense embedding is a blocking Ollama HTTP round-trip; sparse is local
    # CPU work. Run them concurrently rather than paying both latencies
    # back-to-back on every query - this is the hottest path in the system.
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(embed_dense)
        sparse_future = executor.submit(embed_sparse)
        dense_vector = dense_future.result()
        sparse_vector = sparse_future.result()

    prefetch_limit = top_k * _PREFETCH_OVERFETCH_FACTOR

    result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=access_filter,
            ),
            Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=access_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [
        SearchCandidate(
            relative_path=point.payload["relative_path"],
            chunk_index=point.payload["chunk_index"],
            text=point.payload["text"],
            access_tier=point.payload["access_tier"],
            score=point.score,
        )
        for point in result.points
    ]
