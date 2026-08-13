from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.decompose import decompose_query
from agentic_rag.retrieval.rerank import rerank
from agentic_rag.retrieval.search import SearchCandidate, hybrid_search

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
    two separate ones to implement.
    """
    outcomes: list[RetrievalOutcome] = []

    for attempt in range(1, max_attempts + 1):
        sub_questions = decompose_query(
            query,
            model=generation_model,
            base_url=ollama_base_url,
            timeout=generation_timeout_seconds,
        )

        outcomes = []
        for sub_question in sub_questions:
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
            outcomes.append(
                RetrievalOutcome(sub_question=sub_question, candidates=candidates)
            )

        if all(outcome.candidates for outcome in outcomes):
            return PlanningResult(
                sufficient=True, outcomes=outcomes, attempts_used=attempt
            )

    return PlanningResult(
        sufficient=False, outcomes=outcomes, attempts_used=max_attempts
    )
