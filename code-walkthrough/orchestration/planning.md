# `orchestration/planning.py`

**Purpose:** This file is the "retrieval planner" — it takes a user's question, breaks it into smaller sub-questions (via decomposition, handled elsewhere in `decompose.py`), retrieves and reranks the best matching document chunks for each sub-question, and decides whether the overall retrieval was good enough to attempt an answer. If any sub-question comes back completely empty-handed, it retries the whole process from scratch (re-decomposing the question, since retrying the exact same sub-question against the same data would just return the same nothing) — up to a fixed number of attempts, using a progressively more randomized decomposition each retry to give the system an actual chance at a different, better result rather than just repeating a failure. If it still can't find evidence after all attempts, it produces the canonical "I don't know" response the rest of the system relies on.

## Line-by-line walkthrough

### Lines 1-11 — Imports
```python
from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.embedding.ollama_client import EmbeddingError
from agentic_rag.embedding.sparse_client import SparseEmbeddingError
from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.decompose import decompose_query
from agentic_rag.retrieval.rerank import RerankError, rerank
from agentic_rag.retrieval.search import SearchCandidate, hybrid_search
```
- `from dataclasses import dataclass` — used to define the plain result containers below.
- `from qdrant_client import QdrantClient` — the client type for talking to Qdrant, the vector database (a database specialized for searching by semantic similarity rather than exact match) this system stores document embeddings in.
- `from agentic_rag.embedding.cache import EmbeddingCache` — the type for a cache of previously computed embeddings (numeric representations of text), passed through so repeated retrieval work doesn't recompute the same embeddings needlessly.
- `from agentic_rag.embedding.ollama_client import EmbeddingError` and `from agentic_rag.embedding.sparse_client import SparseEmbeddingError` — the specific exception types raised if computing a dense (semantic) or sparse (keyword-style) embedding fails.
- `from agentic_rag.generation.llm_client import GenerationError` — the exception type raised if a call to the LLM (used inside `decompose_query`) fails.
- `from agentic_rag.orchestration.decompose import decompose_query` — the function that breaks a user's question into a list of smaller, more specific sub-questions.
- `from agentic_rag.retrieval.rerank import RerankError, rerank` — `rerank` reorders a list of retrieved candidates by relevance using a cross-encoder model (a model that scores a query and a passage together, generally more accurate but slower than the initial retrieval); `RerankError` is what it raises on failure.
- `from agentic_rag.retrieval.search import SearchCandidate, hybrid_search` — `SearchCandidate` is the type representing one retrieved document chunk; `hybrid_search` is the function that actually searches Qdrant, combining both dense (semantic) and sparse (keyword) search.

### Lines 13-24 — Which errors count as "worth one retry"
```python
# Failures from decompose_query/hybrid_search/rerank that are plausibly
# transient (a one-off bad LLM response, a dropped Ollama connection, a
# reranker scoring hiccup) - these should cost one retry attempt, not
# abort the whole call. UnknownAccessTierError (a bad user_tier) is
# deliberately NOT included: it's a configuration error a retry can never
# fix, so it propagates immediately instead of burning the retry budget.
_TRANSIENT_ATTEMPT_ERRORS = (
    GenerationError,
    EmbeddingError,
    SparseEmbeddingError,
    RerankError,
)
```
- The comment draws a clear line between two categories of failure: transient, likely-temporary problems (a flaky LLM response, a dropped network connection, a scoring hiccup) versus a fundamental configuration mistake (an unrecognized user access tier). The former is worth retrying — a later attempt might just work. The latter can never be fixed by retrying, so it's deliberately *excluded* from this tuple and is allowed to propagate immediately instead of wasting one of the limited retry attempts on a failure that will happen identically every time.
- `_TRANSIENT_ATTEMPT_ERRORS = (GenerationError, EmbeddingError, SparseEmbeddingError, RerankError)` — a tuple of exception types. Python's `except` clause can catch multiple exception types at once by passing a tuple like this, which is used later in the retry loop.

### Lines 26-32 — The shared "can't answer" message
```python
# The single canonical "no answer" message (REQUIREMENTS.md §8 rule 2),
# defined once here since this is the first place that needs it. Reused
# for both the direct-no-match path (insufficient on the very first
# attempt) and the exhausted-retry path (still insufficient after
# max_attempts) - there is exactly one message for "couldn't answer",
# not two, so both paths must produce the identical string.
CANNOT_ANSWER_MESSAGE = "I do not know the answer based on indexed documents."
```
- Defines, once, the exact text used whenever the system genuinely can't find enough evidence to answer. The comment stresses this must be the *single* fallback string used everywhere — not one message for "found nothing on the first try" and a different one for "still found nothing after retrying" — so downstream code (and users) always see identical, predictable wording for the same underlying situation.

### Lines 35-38 — `RetrievalOutcome`
```python
@dataclass(frozen=True)
class RetrievalOutcome:
    sub_question: str
    candidates: list[SearchCandidate]
```
- An immutable container pairing one sub-question with the list of document chunks (`SearchCandidate`s) retrieved (and reranked) for it. `candidates` can be an empty list if nothing relevant was found.

### Lines 41-57 — `PlanningResult` and its validation
```python
@dataclass(frozen=True)
class PlanningResult:
    sufficient: bool
    outcomes: list[RetrievalOutcome]
    attempts_used: int
    message: str | None

    def __post_init__(self) -> None:
        # message is Optional at the type level only because dataclasses
        # can't express "None iff sufficient" directly - callers like
        # generate_answer() rely on message always being a real string
        # when sufficient=False, so a mismatched construction must fail
        # loudly here rather than surface as a silent None downstream.
        if not self.sufficient and self.message is None:
            raise ValueError("message is required when sufficient=False")
        if self.sufficient and self.message is not None:
            raise ValueError("message must be None when sufficient=True")
```
- The overall result of running the planner: `sufficient` (was retrieval good enough to attempt an answer), `outcomes` (the retrieval results for every sub-question tried), `attempts_used` (how many decompose+retrieve attempts it actually took), and `message` (the fallback text, or `None`).
- The comment explains a real limitation of Python's type system: dataclasses can't natively express a rule like "this field must be `None` exactly when that other field is `False`" — the type system alone would let you construct an invalid combination (e.g. `sufficient=False` with `message=None`) with no error. `__post_init__` (a dataclass hook automatically called right after construction) is used to enforce that rule manually, at runtime, so a bug that constructs an inconsistent `PlanningResult` fails loudly and immediately rather than quietly handing a `None` message downstream to code (like `generate_answer()`) that assumes it's always a real string when `sufficient=False`.
- `if not self.sufficient and self.message is None: raise ValueError(...)` — if retrieval was insufficient but no message was supplied, that's an invalid/buggy construction; raise immediately.
- `if self.sufficient and self.message is not None: raise ValueError(...)` — symmetric check: if retrieval succeeded, there should be no leftover fallback message.

### Lines 60-98 — `_retrieve_outcome`: retrieve+rerank for one sub-question
```python
def _retrieve_outcome(
    sub_question: str,
    *,
    client: QdrantClient,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout_seconds: int,
    sparse_model: str,
    embedding_cache: EmbeddingCache,
    reranker_model: str,
    user_tier: str,
    known_tiers: list[str],
    retrieval_top_k: int,
    rerank_top_k: int,
) -> RetrievalOutcome:
    """Retrieve+rerank evidence for a single sub-question - the retryable
    unit `plan_and_retrieve` runs once per sub-question per attempt."""
    candidates = hybrid_search(
        client,
        collection_name,
        sub_question,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        embedding_timeout_seconds=embedding_timeout_seconds,
        sparse_model=sparse_model,
        embedding_cache=embedding_cache,
        user_tier=user_tier,
        known_tiers=known_tiers,
        top_k=retrieval_top_k,
    )
    if candidates:
        candidates = rerank(
            sub_question,
            candidates,
            model_name=reranker_model,
            top_k=rerank_top_k,
        )
    return RetrievalOutcome(sub_question=sub_question, candidates=candidates)
```
- A private helper function (the leading underscore signals it's internal to this module, not part of its public interface). It's named with the singular "outcome" because it handles exactly one sub-question at a time — the docstring calls it "the retryable unit," meaning this is the piece of work the outer retry loop repeats.
- The long parameter list is entirely configuration and infrastructure objects (the Qdrant client, collection name, embedding model settings, the reranker model name, the user's access tier and the full list of known tiers, and the top-k limits controlling how many results to keep at each stage) passed through from the caller — this function itself makes no decisions about these values, it just forwards them.
- `candidates = hybrid_search(...)` — runs the actual search against Qdrant for this sub-question, combining dense (semantic/meaning-based) and sparse (keyword-based) search, filtered to only documents the user's tier is allowed to see, returning up to `retrieval_top_k` candidates.
- `if candidates: candidates = rerank(...)` — only bothers reranking if there's anything to rerank in the first place (skips wasted work on an empty list). Reranking uses a more accurate but more expensive cross-encoder model to reorder and trim the candidates down to `rerank_top_k`.
- `return RetrievalOutcome(sub_question=sub_question, candidates=candidates)` — packages the (possibly reranked, possibly still empty) results together with the sub-question they came from.

### Lines 101-121 — `plan_and_retrieve` signature
```python
def plan_and_retrieve(
    client: QdrantClient,
    collection_name: str,
    query: str,
    *,
    embedding_model: str,
    ollama_base_url: str,
    embedding_timeout_seconds: int,
    sparse_model: str,
    embedding_cache: EmbeddingCache,
    reranker_model: str,
    generation_model: str,
    generation_timeout_seconds: int,
    user_tier: str,
    known_tiers: list[str],
    retrieval_top_k: int,
    rerank_top_k: int,
    max_attempts: int,
    decompose_temperature: float,
    decompose_retry_temperature: float,
) -> PlanningResult:
```
- The public entry point. Takes the raw `query`, the Qdrant client/collection to search, all the same infrastructure settings as `_retrieve_outcome` plus a few more specific to this outer function: `generation_model`/`generation_timeout_seconds` (used by `decompose_query`, which itself calls an LLM to split the question), `max_attempts` (the retry ceiling), and two separate temperature settings — `decompose_temperature` and `decompose_retry_temperature` — whose distinct purposes are explained in the long docstring below.

### Lines 122-184 — The docstring: what "sufficient" means and why the design looks this way
The docstring is long and explains several deliberate design decisions:
- **What "sufficient" means:** every sub-question has at least one candidate chunk after reranking. It's explicitly called a coarse, retrieval-only signal (some evidence vs. none at all) — not a judgment of whether the eventual generated answer will actually be *good*, because that quality can't be known until generation actually happens (a separate, later stage of the pipeline). If even one sub-question comes back with zero evidence, the *whole* attempt is retried by re-decomposing the query from scratch — not just retrying the same failed sub-question — because for a fixed, deterministic dataset and embedding model, retrying the identical sub-question against the same index would return the exact same nothing; only getting a *different* decomposition (different phrasing of the sub-questions) has any real chance at a different, better outcome.
- **Why two different decompose temperatures, not one:** this is explained as a deliberate fix for a subtle bug found during self-review. `generate_answer()` elsewhere in the codebase had its LLM temperature pinned to `0.0` for full determinism, and `decompose_query()` had the identical "unpinned temperature" problem — but naively pinning *this* function's decomposition to `0.0` everywhere would have silently defeated the entire purpose of this retry loop: a fully deterministic `decompose_query()` retried with the exact same input would just reproduce the exact same "insufficient" result forever for any query that only fails because of *how* it happened to get split into sub-questions. So the design keeps attempt 1 low/deterministic (`decompose_temperature`) — because live testing showed identical calls at Ollama's default (higher, unpinned) temperature could produce different sub-question pairs and sometimes wrongly judge a perfectly retrievable query as insufficient purely by chance — while every retry after that deliberately uses a higher `decompose_retry_temperature`, to actually explore different phrasings rather than just repeating the same roll. This turns what used to be accidental, undocumented randomness into an intentional, purposeful retry strategy.
- **Why there's no fixed cutoff on the reranker's relevance score:** the docstring documents that this was tried and rejected, with a concrete live-tested example: a genuinely relevant candidate for "Who played for Arsenal against Chelsea?" scored -5.88 on the cross-encoder, which was actually *worse* (lower) than a genuinely irrelevant candidate for "What is the name of the capital of France?" at -4.44. Because relevant and irrelevant score ranges overlap too much for short, generically phrased questions, any single global cutoff would either wrongly drop real evidence or wrongly let noise through depending on the specific query — a worse signal than the simple "found something vs. found nothing" check this function actually uses. Real judgment about whether retrieved text actually *answers* the question needs an LLM to reason over the text itself, which is a job for the generation stage, not a retrieval-time score threshold.
- **Why "no evidence on attempt 1" and "still no evidence after max_attempts" share one code path:** both situations need the identical `sufficient=False` fallback behavior, so the code doesn't implement them as two separate branches — they're naturally the same path (the loop simply runs out without ever finding success), and `message` carries `CANNOT_ANSWER_MESSAGE` for that one shared case.
- **Why a single failed attempt (an exception) is treated the same as "found no evidence":** if `decompose_query`, `hybrid_search`, or `rerank` raises an exception during one attempt (e.g. a dropped Ollama connection), that costs exactly one retry, the same as if the attempt had run successfully but found nothing — because the failure is plausibly transient and a later attempt may well succeed. The one deliberate exception is `UnknownAccessTierError` (a bad `user_tier`), which is a configuration error no retry can ever fix, so it's allowed to propagate immediately rather than burning through the retry budget on a guaranteed repeat failure.

### Lines 185-197 — Starting the retry loop and running decomposition
```python
    outcomes: list[RetrievalOutcome] = []

    for attempt in range(1, max_attempts + 1):
        try:
            sub_questions = decompose_query(
                query,
                model=generation_model,
                base_url=ollama_base_url,
                timeout=generation_timeout_seconds,
                temperature=(
                    decompose_temperature if attempt == 1 else decompose_retry_temperature
                ),
            )
```
- `outcomes: list[RetrievalOutcome] = []` — initialized before the loop so it has a sensible (empty) value available even if every attempt fails, since it's referenced again after the loop exits.
- `for attempt in range(1, max_attempts + 1):` — loops through attempt numbers `1, 2, ..., max_attempts` (starting at 1 rather than 0 makes the attempt-number comparison below, and any logging/debugging, read naturally as "attempt 1," "attempt 2," etc.).
- `try:` — wraps the whole per-attempt work (decompose + retrieve for every sub-question) so that a transient failure anywhere inside can be caught and treated as "this attempt failed, try again" rather than crashing the whole function.
- `sub_questions = decompose_query(query, model=generation_model, base_url=ollama_base_url, timeout=generation_timeout_seconds, temperature=(decompose_temperature if attempt == 1 else decompose_retry_temperature))` — calls the decomposition function to split the original `query` into a list of sub-question strings. The `temperature` argument is exactly the "two temperatures" design discussed above: attempt 1 uses the low, deterministic `decompose_temperature`; every retry (attempt 2 onward) uses the higher `decompose_retry_temperature` to actually get different phrasing.

### Lines 198-217 — Retrieving for every sub-question, and handling transient failures
```python
            outcomes = [
                _retrieve_outcome(
                    sub_question,
                    client=client,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    ollama_base_url=ollama_base_url,
                    embedding_timeout_seconds=embedding_timeout_seconds,
                    sparse_model=sparse_model,
                    embedding_cache=embedding_cache,
                    reranker_model=reranker_model,
                    user_tier=user_tier,
                    known_tiers=known_tiers,
                    retrieval_top_k=retrieval_top_k,
                    rerank_top_k=rerank_top_k,
                )
                for sub_question in sub_questions
            ]
        except _TRANSIENT_ATTEMPT_ERRORS:
            continue
```
- The list comprehension calls `_retrieve_outcome()` once for every sub-question returned by decomposition, building the full list of `RetrievalOutcome`s for this attempt — this is where each sub-question actually gets searched and reranked, using all the same infrastructure settings passed straight through.
- `except _TRANSIENT_ATTEMPT_ERRORS: continue` — if decomposition or any retrieval/rerank call inside this `try` block raised one of the exception types deemed transient (defined at the top of the file), the loop simply moves on to the next `attempt` iteration rather than letting the exception crash the whole function. `UnknownAccessTierError`, not being in that tuple, is deliberately *not* caught here, so it propagates up immediately as intended.

### Lines 219-232 — Deciding sufficiency and returning the result
```python
        if all(outcome.candidates for outcome in outcomes):
            return PlanningResult(
                sufficient=True,
                outcomes=outcomes,
                attempts_used=attempt,
                message=None,
            )

    return PlanningResult(
        sufficient=False,
        outcomes=outcomes,
        attempts_used=max_attempts,
        message=CANNOT_ANSWER_MESSAGE,
    )
```
- `if all(outcome.candidates for outcome in outcomes):` — checks whether *every* sub-question's outcome has a non-empty `candidates` list (an empty list is falsy in Python, so `all(...)` here means "every outcome found at least one candidate"). This is only reached if the `try` block above completed without a transient exception.
- If so, returns immediately with `sufficient=True`, the full set of `outcomes`, `attempts_used=attempt` (recording exactly how many tries it took), and `message=None` (satisfying the `PlanningResult.__post_init__` rule that `message` must be `None` when `sufficient=True`).
- If the `if` doesn't trigger (some sub-question came back empty), the loop simply continues to the next attempt with a fresh decomposition.
- If the loop finishes all `max_attempts` without ever returning early, execution falls through to the final `return` statement: `sufficient=False`, whatever `outcomes` were computed on the very last attempt, `attempts_used=max_attempts` (the retry budget was fully exhausted), and `message=CANNOT_ANSWER_MESSAGE` — the single canonical "I don't know" string defined at the top of the file, satisfying the other half of the `__post_init__` validation rule.
