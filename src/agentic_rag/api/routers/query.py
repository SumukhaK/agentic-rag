from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import QdrantClient

from agentic_rag.api.dependencies import (
    get_embedding_cache,
    get_qdrant_client,
    get_semantic_cache,
    get_settings,
)
from agentic_rag.api.schemas import CitationModel, QueryRequest, QueryResponse
from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.foul_language import (
    FOUL_LANGUAGE_REFUSAL_MESSAGE,
    check_for_foul_language,
)
from agentic_rag.orchestration.injection_judge import check_for_injection
from agentic_rag.orchestration.output_security import check_output_security
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE
from agentic_rag.orchestration.rewrite import ConversationTurn, rewrite_query
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache
from agentic_rag.retrieval.access import UnknownAccessTierError

router = APIRouter()


def _screen_input(query: str, *, settings: Settings) -> str | None:
    """Screen `query` for a prompt injection attempt and for foul/abusive
    language before it's used anywhere else - checked *before*
    `rewrite_query()` runs at all, since `rewrite_query()` makes its own
    LLM call that the raw query would otherwise reach unchecked.

    Both checks run concurrently via a thread pool - the same pattern
    `hybrid_search()` (`retrieval/search.py`) already established for
    independent Ollama-backed work on the hot path, since neither check
    depends on the other's result.

    Returns the refusal message to send back verbatim if either check
    flags the query, `None` if both are clean. Injection reuses the
    single canonical `CANNOT_ANSWER_MESSAGE` (REQUIREMENTS.md §8 rule 2)
    rather than a distinct message, deliberately, so a would-be attacker
    can't tell an injection attempt was specifically what triggered a
    refusal versus any other reason the system declined to answer. Foul
    language gets its own distinct `FOUL_LANGUAGE_REFUSAL_MESSAGE`
    instead - see `foul_language.py`'s docstring for why that one isn't
    an adversarial-calibration risk the same way.

    Only the current turn's `query` is screened, not the conversation
    history the client resent alongside it - REQUIREMENTS.md §12 says
    "incoming user queries," and history screening (each prior turn was
    already screened as *its own* current query when it was first sent)
    is a separate concern this doesn't attempt to solve.
    """
    judge_kwargs = dict(
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        timeout=settings.generation_timeout_seconds,
        temperature=settings.judge_temperature,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        injection_future = executor.submit(check_for_injection, query, **judge_kwargs)
        foul_language_future = executor.submit(check_for_foul_language, query, **judge_kwargs)
        injection_result = injection_future.result()
        foul_language_result = foul_language_future.result()

    if injection_result.is_injection:
        return CANNOT_ANSWER_MESSAGE
    if foul_language_result.is_foul:
        return FOUL_LANGUAGE_REFUSAL_MESSAGE
    return None


QUERY_422_DESCRIPTION = (
    "Two distinct failure shapes share this status code: a request "
    "validation failure (`HTTPValidationError` below - a `detail` array of "
    "field errors, e.g. an empty `query`), or an unrecognized `user_tier` "
    "that passed request validation but isn't a known access tier "
    "(`{\"detail\": \"<message>\"}` - a plain string, not an array). "
    "`api/app.py`'s custom `openapi()` rewrites this response's default "
    "\"Validation Error\" description to this text - see its docstring "
    "for why a route-level `responses={422: ...}` override can't do this "
    "safely."
)


@router.post("/query", response_model=QueryResponse, summary="Answer a grounded football question")
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
    `citations` resolves every `[N]` marker in `answer` to its actual
    source (`relative_path`/`chunk_index`/`access_tier`) - see
    `AnswerResult` (`orchestration/answer.py`) for why this is a separate
    field rather than requiring the caller to parse `answer` itself.

    All three security judges (REQUIREMENTS.md §12) are composed in:
    `_screen_input()` checks the raw query for a prompt injection attempt
    and for foul/abusive language before anything else runs.
    `check_output_security()` checks the generated answer - both for an
    access-tier leak (deterministic) and for signs of a successful
    injection reflected in the output (LLM-based) - before it's returned;
    a flagged answer is replaced with the canonical fallback and an empty
    citation list, the same "don't reveal which check caught it" reasoning
    `_screen_input()` uses for the injection case. `check_output_security`
    is checked against the *rewritten* query, not the raw one - it's
    asking "does this answer make sense for what was actually retrieved
    against," which is the rewritten, self-contained question, not
    whatever ambiguous phrasing the user originally typed.

    A `GenerationError` (Ollama unreachable, etc.) is not caught here and
    surfaces as FastAPI's default 500 - structured error responses are an
    open item, not yet specified anywhere in docs/REQUIREMENTS.md. An
    unknown `user_tier` (`UnknownAccessTierError`) *is* caught, since it's
    a client input error, not an infrastructure failure - returned as 422.
    Both `answer_with_cache()` and `check_output_security()` can raise it
    (each calls `allowed_tiers_for()` independently), so both run inside
    the same `try` - a tier valid enough for `known_tiers` to have changed
    since a matching answer was cached could otherwise reach
    `check_output_security()` unvalidated on a cache hit and raise there,
    uncaught.

    `check_output_security()` is skipped entirely when `answer.text` is
    the canonical fallback - `generate_answer()` never attaches citations
    to it, so there is nothing to tier-check, and it's a fixed, known-safe
    string that can't "reflect a successful injection" - calling the judge
    on it would be a pure wasted LLM round-trip.
    """
    refusal = _screen_input(payload.query, settings=settings)
    if refusal is not None:
        return QueryResponse(answer=refusal, citations=[])

    history = [ConversationTurn(t.user_query, t.assistant_answer) for t in payload.history]
    rewritten_query = rewrite_query(
        history,
        payload.query,
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        timeout=settings.generation_timeout_seconds,
        temperature=settings.rewrite_temperature,
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
            generation_temperature=settings.generation_temperature,
            decompose_temperature=settings.decompose_temperature,
            decompose_retry_temperature=settings.decompose_retry_temperature,
            known_tiers=settings.access_tiers,
            retrieval_top_k=settings.retrieval_top_k_candidates,
            rerank_top_k=settings.rerank_top_k,
            max_attempts=settings.max_retrieval_attempts,
            similarity_threshold=settings.semantic_cache_similarity_threshold,
            ttl_seconds=settings.semantic_cache_ttl_seconds,
        )

        if answer.text == CANNOT_ANSWER_MESSAGE:
            return QueryResponse(answer=CANNOT_ANSWER_MESSAGE, citations=[])

        security_result = check_output_security(
            rewritten_query,
            answer.text,
            [citation.access_tier for citation in answer.citations],
            payload.user_tier,
            settings.access_tiers,
            model=settings.generation_model,
            base_url=settings.ollama_base_url,
            timeout=settings.generation_timeout_seconds,
            temperature=settings.judge_temperature,
        )
    except UnknownAccessTierError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not security_result.is_safe:
        return QueryResponse(answer=CANNOT_ANSWER_MESSAGE, citations=[])

    return QueryResponse(
        answer=answer.text,
        citations=[CitationModel(**asdict(citation)) for citation in answer.citations],
    )
