import time
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
from agentic_rag.observability.request_log import (
    VERDICT_ANSWERED,
    VERDICT_CANNOT_ANSWER,
    VERDICT_ERROR,
    VERDICT_REFUSED_FOUL_LANGUAGE,
    VERDICT_REFUSED_INJECTION,
    VERDICT_REFUSED_OUTPUT_SECURITY,
    log_query_request,
)
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


def _screen_input(query: str, *, settings: Settings) -> tuple[str, str] | None:
    """Screen `query` for a prompt injection attempt and for foul/abusive
    language before it's used anywhere else - checked *before*
    `rewrite_query()` runs at all, since `rewrite_query()` makes its own
    LLM call that the raw query would otherwise reach unchecked.

    Both checks run concurrently via a thread pool - the same pattern
    `hybrid_search()` (`retrieval/search.py`) already established for
    independent Ollama-backed work on the hot path, since neither check
    depends on the other's result.

    Returns `(refusal_message, verdict)` if either check flags the
    query, `None` if both are clean. Injection reuses the single
    canonical `CANNOT_ANSWER_MESSAGE` (REQUIREMENTS.md §8 rule 2) rather
    than a distinct message, deliberately, so a would-be attacker can't
    tell an injection attempt was specifically what triggered a refusal
    versus any other reason the system declined to answer. Foul language
    gets its own distinct `FOUL_LANGUAGE_REFUSAL_MESSAGE` instead - see
    `foul_language.py`'s docstring for why that one isn't an
    adversarial-calibration risk the same way. `verdict` is returned
    alongside the message (rather than the caller re-deriving it by
    comparing the message text back against `CANNOT_ANSWER_MESSAGE`) so
    the request log always names the real reason a query was refused,
    even though the two paths deliberately return an identical message.

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
        return CANNOT_ANSWER_MESSAGE, VERDICT_REFUSED_INJECTION
    if foul_language_result.is_foul:
        return FOUL_LANGUAGE_REFUSAL_MESSAGE, VERDICT_REFUSED_FOUL_LANGUAGE
    return None


# The public-facing 422 response description for this route, applied onto
# FastAPI's auto-generated schema by app.py's custom openapi() override
# (see its docstring for why a route-level `responses={422: ...}` override
# can't apply this safely). Kept to only what an API consumer needs to
# know about the two response shapes, not the implementation history of
# how this text gets attached to the schema.
QUERY_422_DESCRIPTION = (
    "Two distinct failure shapes share this status code: a request "
    "validation failure (`HTTPValidationError` below - a `detail` array of "
    "field errors, e.g. an empty `query`), or an unrecognized `user_tier` "
    "that passed request validation but isn't a known access tier "
    "(`{\"detail\": \"<message>\"}` - a plain string, not an array)."
)


@router.post(
    "/query", response_model=QueryResponse, summary="Answer a grounded football question"
)
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

    One structured JSON log line (`observability/request_log.py`) is
    emitted for every request, timed per phase via `time.monotonic()` -
    including a request that raises: the whole body below runs inside one
    `try`, and an exception other than `UnknownAccessTierError` (a
    `GenerationError` from an unreachable/timed-out Ollama call, most
    plausibly) is logged with `VERDICT_ERROR` and whatever partial
    timings/outcome fields were already computed, then re-raised - the
    exact scenario this log exists to help diagnose (an Ollama outage)
    must not be the one scenario it stays silent for. `UnknownAccessTierError`
    is the one case that isn't logged at all: a client input-validation
    failure (422) before any pipeline outcome exists to log, not one of
    the fixed `VERDICT_*` vocabulary's cases.
    """
    request_start = time.monotonic()
    history_turns = len(payload.history)
    timings: dict[str, float] = {}
    rewritten_query: str | None = None
    retrieval_hit_count = 0
    cited_paths: list[str] = []

    def _log(verdict: str) -> None:
        timings["total"] = time.monotonic() - request_start
        log_query_request(
            user_tier=payload.user_tier,
            query=payload.query,
            rewritten_query=rewritten_query,
            history_turns=history_turns,
            verdict=verdict,
            retrieval_hit_count=retrieval_hit_count,
            cited_paths=cited_paths,
            timings_seconds=timings,
        )

    try:
        screen_start = time.monotonic()
        screened = _screen_input(payload.query, settings=settings)
        timings["screen_input"] = time.monotonic() - screen_start
        if screened is not None:
            refusal, verdict = screened
            _log(verdict)
            return QueryResponse(answer=refusal, citations=[])

        history = [ConversationTurn(t.user_query, t.assistant_answer) for t in payload.history]
        rewrite_start = time.monotonic()
        rewritten_query = rewrite_query(
            history,
            payload.query,
            model=settings.generation_model,
            base_url=settings.ollama_base_url,
            timeout=settings.generation_timeout_seconds,
            temperature=settings.rewrite_temperature,
        )
        timings["rewrite"] = time.monotonic() - rewrite_start

        answer_start = time.monotonic()
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
        timings["answer"] = time.monotonic() - answer_start

        retrieval_hit_count = len(answer.citations)
        cited_paths = [citation.relative_path for citation in answer.citations]

        if answer.text == CANNOT_ANSWER_MESSAGE:
            # The canonical fallback never carries citations
            # (generate_answer()'s contract) - reset to 0/[] rather than
            # trust that invariant silently, so this log field can never
            # drift from what the caller actually received here.
            retrieval_hit_count = 0
            cited_paths = []
            _log(VERDICT_CANNOT_ANSWER)
            return QueryResponse(answer=CANNOT_ANSWER_MESSAGE, citations=[])

        security_start = time.monotonic()
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
        timings["output_security"] = time.monotonic() - security_start
    except UnknownAccessTierError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        _log(VERDICT_ERROR)
        raise

    if not security_result.is_safe:
        # retrieval_hit_count/cited_paths reflect what was actually
        # retrieved and suppressed, not the empty citation list the
        # caller receives - a reader debugging *why* output security
        # flagged this answer needs to see what it flagged, not what got
        # returned instead.
        _log(VERDICT_REFUSED_OUTPUT_SECURITY)
        return QueryResponse(answer=CANNOT_ANSWER_MESSAGE, citations=[])

    _log(VERDICT_ANSWERED)

    return QueryResponse(
        answer=answer.text,
        citations=[CitationModel(**asdict(citation)) for citation in answer.citations],
    )
