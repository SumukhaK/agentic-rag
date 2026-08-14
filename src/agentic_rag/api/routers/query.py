from fastapi import APIRouter, Depends, HTTPException
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
from agentic_rag.retrieval.access import UnknownAccessTierError

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
    conversation turns (FR1/FR2). Stateless: the caller resends the whole
    conversation history every call - see docs/REQUIREMENTS.md §13 for why.
    Citations are embedded in the answer text by `generate_answer()`'s own
    grounding prompt, not returned as a separate field - see
    PROJECT_TRACKER.md's Phase 7 log for the gap this leaves for API
    clients trying to resolve a citation to an actual document, tracked as
    a follow-up rather than fixed here.

    Security judges (`check_for_injection`, `check_for_foul_language`,
    `check_output_security`) are deliberately not composed in yet - see
    PROJECT_TRACKER.md's Phase 7 log.

    A `GenerationError` (Ollama unreachable, etc.) is not caught here and
    surfaces as FastAPI's default 500 - structured error responses are an
    open item, not yet specified anywhere in docs/REQUIREMENTS.md. An
    unknown `user_tier` (`UnknownAccessTierError`) *is* caught, since it's
    a client input error, not an infrastructure failure - returned as 422.
    """
    history = [ConversationTurn(t.user_query, t.assistant_answer) for t in payload.history]
    rewritten_query = rewrite_query(
        history,
        payload.query,
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        timeout=settings.generation_timeout_seconds,
    )

    try:
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
    except UnknownAccessTierError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return QueryResponse(answer=answer)
