# `api/routers/query.py`

**Purpose:** This file defines the single most important endpoint in the whole application: `POST /query`, the one that actually answers a user's football question. It's the place where all the pieces of the system come together in the right order: screening the incoming question for prompt-injection attempts and foul language, rewriting it into a self-contained question using the conversation history, running retrieval-augmented generation (with a semantic cache to avoid redundant work), screening the generated answer for security problems before it's returned, and logging a structured record of everything that happened for observability. Getting this ordering and error-handling right is what makes the system "grounded" and "safe" per the project's AI philosophy — every step here exists to prevent a specific failure mode.

## Line-by-line walkthrough

### Lines 1-38 — Imports and router setup
```python
import time
import uuid
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
```
- `import time` — imports Python's `time` module, used via `time.monotonic()` to measure how long each phase of handling a request takes, for logging.
- `import uuid` — imports Python's UUID-generation module, used to mint a unique `request_id` for every incoming request so its log line can be told apart from any other request's.
- `from concurrent.futures import ThreadPoolExecutor` — imports a tool for running a small, fixed number of tasks concurrently on separate threads, used here to run the injection check and the foul-language check at the same time instead of one after another.
- `from dataclasses import asdict` — imports a helper that converts a Python "dataclass" instance (a simple class for holding data) into a plain dictionary, used later to convert internal citation objects into the API's `CitationModel` schema.
- `from fastapi import APIRouter, Depends, HTTPException` — imports `APIRouter` (to define this file's routes), `Depends` (dependency injection), and `HTTPException` (used to deliberately return a specific HTTP error status, like 422, with a custom message).
- `from qdrant_client import QdrantClient` — imports the Qdrant client type, used as a type hint for the injected dependency.
- `from agentic_rag.api.dependencies import (...)` — imports the four dependency-provider functions this route needs: the embedding cache, the Qdrant client, the semantic cache, and the settings.
- `from agentic_rag.api.schemas import CitationModel, QueryRequest, QueryResponse` — imports the request/response Pydantic models defined in `api/schemas.py`.
- `from agentic_rag.config import Settings` — imports the `Settings` type hint.
- `from agentic_rag.embedding.cache import EmbeddingCache` — imports the embedding cache type hint.
- `from agentic_rag.observability.request_log import (...)` — imports a set of `VERDICT_*` constants (fixed strings naming every possible outcome a request can have, like "answered" or "refused due to injection") and `log_query_request`, the function that actually writes the structured log line, all from `observability/request_log.py`.
- `from agentic_rag.orchestration.foul_language import (...)` — imports `FOUL_LANGUAGE_REFUSAL_MESSAGE` (the fixed text returned when foul language is detected) and `check_for_foul_language` (the function that runs that check).
- `from agentic_rag.orchestration.injection_judge import check_for_injection` — imports the function that checks whether a query looks like a prompt-injection attempt (an attempt to manipulate the assistant into ignoring its instructions).
- `from agentic_rag.orchestration.output_security import check_output_security` — imports the function that checks a *generated answer* (not the input query) for security problems before it's sent back to the user.
- `from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE` — imports the single canonical fallback message used whenever the system declines to give a real answer, for any reason.
- `from agentic_rag.orchestration.rewrite import ConversationTurn, rewrite_query` — imports `ConversationTurn` (a simple data structure representing one prior turn, used internally, distinct from the API's `ConversationTurnModel`) and `rewrite_query`, the function that turns the current question plus history into one self-contained question.
- `from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache` — imports the `SemanticCache` type hint and `answer_with_cache`, the function that either returns a cached answer or actually runs retrieval-augmented generation and caches the result.
- `from agentic_rag.retrieval.access import UnknownAccessTierError` — imports the exception type raised when a request's `user_tier` doesn't match any tier the server knows about.
- `router = APIRouter()` — creates this file's router, later mounted onto the main app in `api/app.py`.

### Lines 40-87 — `_screen_input`: checking the raw query before anything else runs
```python
def _screen_input(query: str, *, settings: Settings) -> tuple[str, str] | None:
    """Screen `query` for a prompt injection attempt and for foul/abusive
    language before it's used anywhere else - checked *before*
    `rewrite_query()` runs at all, since `rewrite_query()` makes its own
    LLM call that the raw query would otherwise reach unchecked.
    ...
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
```
- `def _screen_input(query: str, *, settings: Settings) -> tuple[str, str] | None:` — defines a private helper function (the leading underscore signals it's internal to this module) that takes the raw query text and the app settings (`*,` forces `settings` to be passed by keyword, preventing an accidental positional mix-up), and returns either `None` (nothing wrong found) or a tuple of `(refusal_message, verdict)`.
- The docstring explains this screens the query for two things — a prompt-injection attempt, and foul/abusive language — and stresses this must happen *before* `rewrite_query()` is called at all, because `rewrite_query()` itself makes a call to the LLM, and an un-screened, potentially malicious query would otherwise reach that LLM call completely unchecked.
- It explains both checks run concurrently using a thread pool, reusing the same concurrency pattern already used elsewhere in the codebase (`hybrid_search()` in `retrieval/search.py`) for independent Ollama-backed calls — since neither check's result depends on the other, running them in parallel is both safe and faster than running them one after another.
- It explains the return value: when either check flags a problem, the function returns a `(message, verdict)` pair. For an injection, it deliberately returns the exact same canonical `CANNOT_ANSWER_MESSAGE` used for every other "can't answer" situation, rather than a distinct "this looks like an injection attempt" message — specifically so that someone attempting an injection attack can't tell from the response alone whether their attempt was detected as an injection versus refused for any other reason, which would otherwise leak information useful for calibrating further attack attempts. Foul language, by contrast, does get its own distinct message (`FOUL_LANGUAGE_REFUSAL_MESSAGE`) because it isn't considered the same kind of adversarial-calibration risk. The `verdict` string is returned alongside the message so that the request log can record the real, specific reason a query was refused, even though the message text shown to the caller is deliberately identical either way.
- It also notes that only the *current* turn's query is screened here — not the conversation history resent by the client — because each prior turn was already screened as its own "current query" back when it was originally submitted; screening history again is explicitly out of scope for this function.
- `judge_kwargs = dict(model=settings.generation_model, base_url=settings.ollama_base_url, timeout=settings.generation_timeout_seconds, temperature=settings.judge_temperature)` — bundles the common configuration both checks need (which LLM model to use, where Ollama is running, how long to wait, and how deterministic to make the output) into one dictionary, so it only has to be written once.
- `with ThreadPoolExecutor(max_workers=2) as executor:` — opens a thread pool limited to 2 worker threads (exactly enough for the two checks below), automatically cleaned up when the `with` block ends.
- `injection_future = executor.submit(check_for_injection, query, **judge_kwargs)` — schedules the injection check to run on a thread, immediately returning a "future" (a placeholder representing a result that isn't ready yet) rather than blocking.
- `foul_language_future = executor.submit(check_for_foul_language, query, **judge_kwargs)` — schedules the foul-language check to run on another thread at the same time.
- `injection_result = injection_future.result()` — blocks until the injection check's thread finishes, then retrieves its result.
- `foul_language_result = foul_language_future.result()` — blocks until the foul-language check's thread finishes and retrieves its result (since the injection check was likely already running concurrently, this doesn't add significant extra wait time compared to running them sequentially).
- `if injection_result.is_injection: return CANNOT_ANSWER_MESSAGE, VERDICT_REFUSED_INJECTION` — if the injection judge flagged the query, returns the shared fallback message paired with the specific injection verdict.
- `if foul_language_result.is_foul: return FOUL_LANGUAGE_REFUSAL_MESSAGE, VERDICT_REFUSED_FOUL_LANGUAGE` — if the foul-language judge flagged the query (checked second, so an injection verdict takes priority if somehow both were flagged), returns the distinct foul-language message and verdict.
- `return None` — if neither check flagged anything, signals to the caller that the query is clean and processing should continue normally.

### Lines 90-102 — The shared 422 description constant
```python
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
```
- The comment explains this constant exists specifically to be imported by `api/app.py`'s custom OpenAPI-schema patch (discussed in that file's own walkthrough), rather than being applied directly on this route's decorator, because a direct `responses={422: ...}` override here would silently break FastAPI's own auto-generated 422 schema entry instead of extending it. It also clarifies the text is written for an external API consumer to understand the two possible 422 response shapes, not to document how the patching mechanism itself works internally.
- `QUERY_422_DESCRIPTION = (...)` — the actual text: it explains that a 422 status from this endpoint can mean one of two different things — either a standard request-validation failure (like sending a blank `query`, producing FastAPI's normal `HTTPValidationError` array-of-errors shape), or a syntactically valid request whose `user_tier` simply isn't one of the tiers the server recognizes (producing a much simpler shape: a plain string inside a `detail` key, not an array).

### Lines 105-114 — The route decorator and function signature
```python
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
```
- `@router.post("/query", response_model=QueryResponse, summary="Answer a grounded football question")` — registers this function as the handler for `POST /query`, declares its successful response shape, and gives it a documentation summary.
- `def query(payload: QueryRequest, ...)` — the handler function; `payload: QueryRequest` is the parsed and validated request body (FastAPI automatically parses the incoming JSON into this Pydantic model and rejects anything that doesn't validate, before this function is even called).
- `settings`, `client`, `embedding_cache`, `cache` — the four shared resources needed to answer a query, all supplied via dependency injection from `api/dependencies.py` rather than being constructed or imported directly here, which is what makes this route testable with fake/temporary versions of each.

### Lines 115-167 — The route's docstring
```python
    """Answer `payload.query` for `payload.user_tier`, given prior
    conversation turns (FR1/FR2). Stateless: the caller resends the whole
    conversation history every call - see docs/REQUIREMENTS.md §13 for why.
    ...
    """
```
- The docstring lays out the endpoint's overall contract and reasoning, matched to the code below it:
  - It answers `payload.query` for `payload.user_tier`, given the resent conversation history (referencing functional requirements FR1/FR2), and reiterates the endpoint is stateless — the caller must resend the full history each time, per `docs/REQUIREMENTS.md` §13.
  - `citations` resolves every `[N]` marker in the answer text back to its real source, rather than requiring the caller to somehow parse the answer text themselves to figure out what was cited — that separation is explained further in `orchestration/answer.py`'s own documentation of `AnswerResult`.
  - It explains how the three required security judges (per requirements §12) are composed together: `_screen_input()` checks the raw query first (injection + foul language); `check_output_security()` checks the *generated answer* afterward, for two different things — whether it leaks content from an access tier the requesting user isn't allowed to see (a deterministic check), and whether it shows signs that an injection attempt succeeded and got reflected into the output (an LLM-based check). If flagged, the answer is replaced with the same canonical fallback and an empty citation list — following the same "don't reveal which specific check caught it" reasoning used for the input-injection case. Critically, this output check is run against the *rewritten* query, not the user's original raw phrasing, because the real question being asked is "does this answer make sense for what was actually retrieved against" — and what was actually retrieved against is the rewritten, self-contained question, not whatever ambiguous or context-dependent phrasing the user originally typed.
  - It explains error handling: an infrastructure-level failure like `GenerationError` (for example, Ollama being unreachable) is deliberately *not* caught here, and is allowed to surface as FastAPI's default 500 response — because structured error responses for that case are explicitly called out as an open item, not yet specified in the requirements document. By contrast, `UnknownAccessTierError` — a client input mistake, not an infrastructure failure — *is* caught and turned into a 422. It explains that both `answer_with_cache()` and `check_output_security()` can independently raise `UnknownAccessTierError` (each calls `allowed_tiers_for()` on its own), so both calls must be inside the same `try` block — otherwise, a tier that seemed valid enough to be served from the semantic cache, if the set of known tiers had since changed, could reach `check_output_security()` completely unvalidated and raise there uncaught.
  - It explains `check_output_security()` is skipped entirely whenever the answer text is exactly the canonical fallback message: `generate_answer()` never attaches real citations to that fallback, so there's nothing meaningful to tier-check, and the fallback text itself is a fixed, already-known-safe string that can't possibly "reflect a successful injection" — so calling the security judge on it would just waste an LLM round-trip for no benefit.
  - It explains the observability behavior: exactly one structured JSON log line is emitted per request (via `log_query_request` from `observability/request_log.py`), with each phase's duration measured using `time.monotonic()`. This happens *even for a request that raises an exception* — the entire body of the function runs inside one `try` block, and any exception other than `UnknownAccessTierError` (most plausibly a `GenerationError` from Ollama being unreachable or timing out) is logged with the `VERDICT_ERROR` verdict, along with whatever partial timing/outcome data had already been computed before the failure, and then the exception is re-raised — deliberately, so that the exact scenario this logging exists to help diagnose (an Ollama outage) is never the one scenario that goes unlogged. The one genuine exception to "always log something" is `UnknownAccessTierError` itself, which represents a client input-validation failure (a 422) that happens before there's any real pipeline outcome to log, and doesn't correspond to any of the fixed `VERDICT_*` vocabulary anyway.

### Lines 169-175 — Setting up per-request tracking state
```python
    request_id = str(uuid.uuid4())
    request_start = time.monotonic()
    history_turns = len(payload.history)
    timings: dict[str, float] = {}
    rewritten_query: str | None = None
    retrieval_hit_count = 0
    cited_paths: list[str] = []
```
- `request_id = str(uuid.uuid4())` — the very first thing this function does, before any real work happens: mints a fresh, random UUID and converts it to its standard string form (e.g. `"3fa85f64-5717-4562-b3fc-2c963f66afa6"`). Generated once per request and reused for the entire life of this one call - this is what lets a reader tell two requests' log lines apart, even if everything else about them (query text, user tier) happens to be identical.
- `request_start = time.monotonic()` — records the moment this request started being processed, used as the baseline for the total-duration measurement (`time.monotonic()` is used instead of wall-clock time because it can't jump backward or be affected by system clock changes, making duration measurements reliable).
- `history_turns = len(payload.history)` — captures how many prior conversation turns were sent, for inclusion in the log line.
- `timings: dict[str, float] = {}` — starts an empty dictionary that will accumulate how long each phase of processing took, filled in as the function progresses.
- `rewritten_query: str | None = None` — starts as `None` and will be filled in once the query has been rewritten; kept as `None` until then so the log accurately reflects that rewriting never happened if, say, input screening rejected the query first.
- `retrieval_hit_count = 0` — starts at zero; will be updated to reflect how many source citations were actually found, if processing gets that far.
- `cited_paths: list[str] = []` — starts empty; will be filled with the relative paths of any cited documents, for the log line.

### Lines 177-188 — The inner `_log` helper
```python
    def _log(verdict: str) -> None:
        timings["total"] = time.monotonic() - request_start
        log_query_request(
            request_id=request_id,
            user_tier=payload.user_tier,
            query=payload.query,
            rewritten_query=rewritten_query,
            history_turns=history_turns,
            verdict=verdict,
            retrieval_hit_count=retrieval_hit_count,
            cited_paths=cited_paths,
            timings_seconds=timings,
        )
```
- `def _log(verdict: str) -> None:` — defines a small closure (a function defined inside another function, which can read and use the outer function's local variables directly) that captures the current values of all the tracking variables set up above and writes one structured log line. Defining it as a closure means every one of the several places in the function below that needs to log a final outcome can just call `_log(some_verdict)` without repeating all this bookkeeping each time.
- `timings["total"] = time.monotonic() - request_start` — computes and records the total elapsed time for the whole request, right before logging.
- `log_query_request(...)` — calls the actual logging function (from `observability/request_log.py`), passing along this request's unique ID, the user's tier, their original query, the rewritten query (or `None` if rewriting never happened), how many history turns were involved, the specific outcome verdict, how many results were retrieved, which document paths were cited, and the full per-phase timing breakdown.

### Lines 188-195 — Screening the input
```python
    try:
        screen_start = time.monotonic()
        screened = _screen_input(payload.query, settings=settings)
        timings["screen_input"] = time.monotonic() - screen_start
        if screened is not None:
            refusal, verdict = screened
            _log(verdict)
            return QueryResponse(answer=refusal, citations=[])
```
- `try:` — opens the single large try block that wraps essentially the entire rest of the function's logic, as described in the docstring, so that any unexpected exception can still be logged before it propagates.
- `screen_start = time.monotonic()` — marks the start of the input-screening phase.
- `screened = _screen_input(payload.query, settings=settings)` — runs the injection/foul-language checks defined earlier on the raw incoming query.
- `timings["screen_input"] = time.monotonic() - screen_start` — records how long screening took.
- `if screened is not None: refusal, verdict = screened; _log(verdict); return QueryResponse(answer=refusal, citations=[])` — if screening flagged a problem, unpacks the returned `(message, verdict)` pair, logs the outcome, and immediately returns a response carrying the refusal message and no citations — ending processing here without ever reaching retrieval or generation.

### Lines 197-207 — Rewriting the query
```python
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
```
- `history = [ConversationTurn(t.user_query, t.assistant_answer) for t in payload.history]` — converts each API-schema `ConversationTurnModel` entry in the request into the internal `ConversationTurn` data structure that `rewrite_query` expects, using a list comprehension (a compact way of building a new list by transforming each item of an existing one).
- `rewrite_start = time.monotonic()` — marks the start of the rewrite phase.
- `rewritten_query = rewrite_query(history, payload.query, model=..., base_url=..., timeout=..., temperature=...)` — calls the LLM-backed function that turns the current question plus prior turns into one self-contained question (for example, resolving a pronoun like "he" using an earlier turn), storing the result in the previously-`None` `rewritten_query` variable so it's now available both to the rest of this function and to the eventual log line.
- `timings["rewrite"] = time.monotonic() - rewrite_start` — records how long rewriting took.

### Lines 209-220 — Getting the answer, with caching
```python
        answer_start = time.monotonic()
        answer = answer_with_cache(
            rewritten_query,
            payload.user_tier,
            cache=cache,
            client=client,
            collection_name=settings.qdrant_collection_name,
            embedding_cache=embedding_cache,
            known_tiers=settings.access_tiers,
            settings=settings,
        )
        timings["answer"] = time.monotonic() - answer_start
```
- `answer_start = time.monotonic()` — marks the start of the answer-generation phase.
- `answer = answer_with_cache(rewritten_query, payload.user_tier, cache=cache, client=client, collection_name=settings.qdrant_collection_name, embedding_cache=embedding_cache, known_tiers=settings.access_tiers, settings=settings)` — calls into the orchestration layer's semantic-cache-aware answer function, passing the rewritten query, the user's tier (used to filter what they're allowed to retrieve), the shared semantic cache and embedding cache, the Qdrant client and collection name, the list of known valid tiers (used to validate `user_tier` and raise `UnknownAccessTierError` if it's not recognized), and the settings object. This single call either returns a previously cached answer for a sufficiently similar prior query, or runs the full retrieval-augmented generation pipeline and caches the fresh result.
- `timings["answer"] = time.monotonic() - answer_start` — records how long this phase took (which could be very fast on a cache hit, or slow on a full pipeline run).

### Lines 222-233 — Handling the "cannot answer" case
```python
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
```
- `retrieval_hit_count = len(answer.citations)` — updates the tracking variable to the actual number of citations the answer came with.
- `cited_paths = [citation.relative_path for citation in answer.citations]` — builds the list of cited document paths, again using a list comprehension, for the log line.
- `if answer.text == CANNOT_ANSWER_MESSAGE:` — checks whether the answer produced is exactly the canonical fallback message, meaning no grounded answer could actually be found.
- The comment explains why the two tracking variables get explicitly reset to `0`/`[]` here even though they were presumably already empty: rather than silently trusting that `generate_answer()` never attaches citations to the fallback message (an assumption that could quietly become false if that function's behavior changed later), this code makes the invariant explicit, so the logged values can never drift from what the caller genuinely received.
- `retrieval_hit_count = 0` / `cited_paths = []` — the explicit reset described above.
- `_log(VERDICT_CANNOT_ANSWER)` — logs this outcome with its own specific verdict.
- `return QueryResponse(answer=CANNOT_ANSWER_MESSAGE, citations=[])` — returns the fallback response and ends processing here, before ever reaching the output-security check (consistent with the docstring's explanation that checking security on the fixed fallback string would be pointless).

### Lines 235-247 — Checking output security
```python
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
```
- `security_start = time.monotonic()` — marks the start of the output-security-check phase (only reached if a real, non-fallback answer was produced).
- `security_result = check_output_security(rewritten_query, answer.text, [citation.access_tier for citation in answer.citations], payload.user_tier, settings.access_tiers, model=..., base_url=..., timeout=..., temperature=...)` — calls the function that checks the generated answer for two things at once, as the docstring described: whether it leaks material from an access tier above what this user is allowed to see (using the list of access tiers of the cited sources, the user's own tier, and the full list of known tiers), and whether the answer shows signs of a successful prompt-injection reflected into its own text (checked against the rewritten query, not the raw one, per the docstring's reasoning).
- `timings["output_security"] = time.monotonic() - security_start` — records how long this check took.

### Lines 248-252 — Catching the two possible exception types
```python
    except UnknownAccessTierError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        _log(VERDICT_ERROR)
        raise
```
- `except UnknownAccessTierError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc` — catches the specific case of an unrecognized `user_tier` (which, per the docstring, can be raised by either `answer_with_cache()` or `check_output_security()`), and converts it into a proper 422 HTTP error with the exception's message as the detail. `from exc` preserves the original exception as the documented "cause" of the new one, which keeps the full error chain visible in logs/tracebacks rather than hiding it. Notably, this path does not call `_log()` — matching the docstring's explanation that a client input-validation failure isn't one of the pipeline outcomes this logging exists to track.
- `except Exception: _log(VERDICT_ERROR); raise` — catches any other exception (broadly, matching the "must not silently drop the one scenario this logging most needs to catch" reasoning from the docstring), logs it with the generic `VERDICT_ERROR` verdict (capturing whatever partial timing data had accumulated up to that point), and then re-raises the exact same exception with `raise` (with no arguments), letting it propagate onward to become FastAPI's default 500 response, exactly as the docstring specifies.

### Lines 254-261 — Handling a failed output-security check
```python
    if not security_result.is_safe:
        # retrieval_hit_count/cited_paths reflect what was actually
        # retrieved and suppressed, not the empty citation list the
        # caller receives - a reader debugging *why* output security
        # flagged this answer needs to see what it flagged, not what got
        # returned instead.
        _log(VERDICT_REFUSED_OUTPUT_SECURITY)
        return QueryResponse(answer=CANNOT_ANSWER_MESSAGE, citations=[])
```
- `if not security_result.is_safe:` — checked only if no exception occurred above, meaning a real answer and a completed security check both exist; this branch handles the case where that check found a problem.
- The comment clarifies a subtlety: at this point, `retrieval_hit_count` and `cited_paths` still hold the *real* citation data from the (now-suppressed) answer, not the empty list that's about to be sent back to the caller — and that's deliberate, so that anyone reading the log later to understand *why* this particular answer got flagged can see exactly what was retrieved and suppressed, rather than only seeing the empty citation list the caller actually received.
- `_log(VERDICT_REFUSED_OUTPUT_SECURITY)` — logs this specific outcome.
- `return QueryResponse(answer=CANNOT_ANSWER_MESSAGE, citations=[])` — returns the canonical fallback to the caller, withholding the flagged answer entirely, with no citations (following the same "don't reveal what specifically was caught" pattern used for the input-screening refusals).

### Lines 263-268 — The successful path
```python
    _log(VERDICT_ANSWERED)

    return QueryResponse(
        answer=answer.text,
        citations=[CitationModel(**asdict(citation)) for citation in answer.citations],
    )
```
- `_log(VERDICT_ANSWERED)` — reached only if everything above succeeded and the output-security check passed; logs the fully successful outcome.
- `return QueryResponse(answer=answer.text, citations=[CitationModel(**asdict(citation)) for citation in answer.citations])` — builds and returns the final, real response: the generated answer text as-is, and its citations converted from the internal citation dataclass objects into the API's `CitationModel` schema. `asdict(citation)` turns each internal citation object into a plain dictionary of its fields, and `CitationModel(**...)` unpacks that dictionary into keyword arguments to construct the corresponding API-facing model — done inside a list comprehension so every citation in `answer.citations` gets converted the same way.
