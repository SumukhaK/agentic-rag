from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.embedding.ollama_client import EmbeddingError
from agentic_rag.embedding.sparse_client import SparseEmbeddingError
from agentic_rag.generation.llm_client import GenerationError
from agentic_rag.orchestration.decompose import decompose_query
from agentic_rag.retrieval.rerank import RerankError, rerank
from agentic_rag.retrieval.search import SearchCandidate, hybrid_search

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

# The single canonical "no answer" message (REQUIREMENTS.md §8 rule 2),
# defined once here since this is the first place that needs it. Reused
# for both the direct-no-match path (insufficient on the very first
# attempt) and the exhausted-retry path (still insufficient after
# max_attempts) - there is exactly one message for "couldn't answer",
# not two, so both paths must produce the identical string.
CANNOT_ANSWER_MESSAGE = "I do not know the answer based on indexed documents."


@dataclass(frozen=True)
class RetrievalOutcome:
    sub_question: str
    candidates: list[SearchCandidate]


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
    """Decompose `query`, retrieve+rerank evidence for every sub-question,
    and retry (re-decomposing from scratch) up to `max_attempts` times if
    any sub-question comes back with no evidence at all.

    "Sufficient" means every sub-question has at least one candidate chunk
    after reranking - a coarse, retrieval-only signal (no evidence at all
    vs. some evidence), not a judgment of answer quality, since answer
    quality isn't knowable until generation exists (Phase 5). Insufficient
    for even one sub-question triggers a full retry: re-decomposing gives
    the LLM a fresh chance at different phrasing, which is the only thing
    that can plausibly change the result for a deterministic corpus and
    embeddings - retrying the identical sub-question against the same
    index would just return the same nothing.

    `decompose_temperature` (attempt 1) vs. `decompose_retry_temperature`
    (every attempt after) is a deliberate split, not two arbitrary knobs:
    self-review of `generate_answer()`'s temperature fix found
    `decompose_query()` had the identical unpinned-temperature bug, but
    naively pinning it to `0.0` everywhere (matching `generate_answer()`/
    `rewrite_query()`) would have silently defeated this retry loop's own
    stated purpose above - a deterministic `decompose_query()` retried
    with the exact same input just reproduces the exact same
    "insufficient" result, forever, for any query that only fails because
    of *how* it got decomposed. So attempt 1 stays low/deterministic
    (`decompose_temperature`, matching the live-tested need: identical
    calls at Ollama's default temperature produced different sub-question
    pairs, sometimes wrongly judging a retrievable query `sufficient=False`
    by chance) and every retry deliberately uses a higher
    `decompose_retry_temperature` to actually explore different phrasing -
    turning what was accidental, undocumented randomness into an
    intentional retry strategy.

    A fixed cutoff on the cross-encoder's rerank score was tried as a
    tighter "is this actually relevant" signal and rejected: live-tested
    against the reranker, a genuinely relevant candidate ("Who played for
    Arsenal against Chelsea?") scored -5.88, worse than a genuinely
    irrelevant one ("What is the name of the capital of France?") at
    -4.44. Relevant/irrelevant score ranges overlap too much for a global
    threshold to separate them for short, generically-phrased questions,
    so any cutoff either drops real evidence or lets noise through
    depending on the query - worse than the coarse signal it would
    replace. Real answerability judgment needs the LLM to actually reason
    over the retrieved text, which belongs to generation (Phase 5), not a
    retrieval-time score cutoff.

    Reaching `max_attempts` without every sub-question finding evidence,
    and a single (sub-)question already having none on the very first
    attempt, both produce the same `sufficient=False` result - the
    direct-no-match and exhausted-retry paths the fallback message needs
    to cover (PROJECT_TRACKER.md Phase 4) are really one path here, not
    two separate ones to implement. `message` carries `CANNOT_ANSWER_MESSAGE`
    for that shared `sufficient=False` result and is `None` otherwise, so a
    caller never has to duplicate the fallback string itself.

    A single attempt failing outright (`decompose_query`, `hybrid_search`,
    or `rerank` raising - e.g. a dropped Ollama connection) is treated the
    same as that attempt finding no evidence: it costs one retry, not the
    whole call, since the failure is plausibly transient and a later
    attempt may well succeed. `UnknownAccessTierError` is the one
    exception excluded from this: a bad `user_tier` is a configuration
    error no retry can fix, so it propagates immediately instead of
    burning the retry budget.
    """
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
