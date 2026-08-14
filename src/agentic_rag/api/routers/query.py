from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from agentic_rag.api.dependencies import (
    get_embedding_cache,
    get_qdrant_client,
    get_semantic_cache,
    get_settings,
)
from agentic_rag.api.schemas import QueryRequest, QueryResponse
from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.rewrite import ConversationTurn, rewrite_query
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    client: QdrantClient = Depends(get_qdrant_client),
    embedding_cache: EmbeddingCache = Depends(get_embedding_cache),
    cache: SemanticCache = Depends(get_semantic_cache),
) -> QueryResponse:
    """Answer `payload.query` for `payload.user_tier`, given prior
    conversation turns (FR1/FR2 - citations are embedded in the answer
    text itself by `generate_answer()`'s grounding prompt, not a separate
    field).

    Stateless (recorded product decision, Phase 7): the caller resends the
    whole conversation history on every call - this endpoint holds no
    session state of its own, consistent with this codebase's caches
    already being scoped to the process, not to a conversation.

    Security judges (`check_for_injection`, `check_for_foul_language`,
    `check_output_security`) are deliberately not composed in yet - see
    PROJECT_TRACKER.md's Phase 7 log for why this landed as its own,
    later item rather than bundled here.

    A `GenerationError` from `rewrite_query`/`answer_with_cache` (Ollama
    unreachable, etc.) is not caught here and surfaces as FastAPI's
    default 500 - structured error responses are an open item, not yet
    specified anywhere in docs/REQUIREMENTS.md.
    """
    history = [ConversationTurn(t.user_query, t.assistant_answer) for t in payload.history]
    rewritten_query = rewrite_query(
        history,
        payload.query,
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        timeout=settings.generation_timeout_seconds,
    )

    answer = answer_with_cache(
        rewritten_query,
        payload.user_tier,
        cache=cache,
        client=client,
        collection_name=settings.qdrant_collection_name,
        embedding_model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
        embedding_timeout_seconds=settings.embedding_timeout_seconds,
        sparse_model=settings.sparse_embedding_model,
        embedding_cache=embedding_cache,
        reranker_model=settings.reranker_model,
        generation_model=settings.generation_model,
        generation_timeout_seconds=settings.generation_timeout_seconds,
        known_tiers=settings.access_tiers,
        retrieval_top_k=settings.retrieval_top_k_candidates,
        rerank_top_k=settings.rerank_top_k,
        max_attempts=settings.max_retrieval_attempts,
        similarity_threshold=settings.semantic_cache_similarity_threshold,
        ttl_seconds=settings.semantic_cache_ttl_seconds,
    )

    return QueryResponse(answer=answer)
