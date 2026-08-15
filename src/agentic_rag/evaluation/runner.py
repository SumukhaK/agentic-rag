from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.evaluation.judge import check_faithfulness
from agentic_rag.evaluation.questions import EvalQuestion, load_questions
from agentic_rag.evaluation.report import (
    EvalQuestionResult,
    EvaluationReport,
    build_report,
    report_to_json_dict,
)
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.ingestion.scheduler import run_sync_cycle
from agentic_rag.orchestration.answer import Citation
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache


def _normalize_path(relative_path: str) -> str:
    # Real relative_path values come from the ingestion pipeline's
    # str(path.relative_to(folder)), which uses os.sep - backslash on
    # Windows. expected_source_paths in the JSON fixture are written with
    # forward slashes for portability/readability. Normalizing both sides
    # to forward slashes before comparing avoids the exact "hardcoded
    # forward-slash literal vs. Windows backslash" test bug already hit
    # twice this session (test_scheduler.py, a live smoke-test script).
    return relative_path.replace("\\", "/")


def _fetch_cited_source_text(
    client: QdrantClient, collection_name: str, citation: Citation
) -> str:
    """Fetch the actual indexed chunk text for `citation`, so the
    faithfulness judge can be shown what was really cited rather than
    just its `relative_path`/`chunk_index` metadata.

    `Citation` (`orchestration/answer.py`) deliberately doesn't carry the
    chunk's text - it's citation metadata for the API response, not a
    retrieval result - so this queries Qdrant directly for the payload
    `index_document()` (`indexing/upsert.py`) already stores there.
    Returns `""` if the point can't be found (a stale citation against a
    reindexed corpus) rather than raising - the faithfulness judge will
    then correctly see nothing to support the claim.
    """
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="relative_path", match=MatchValue(value=citation.relative_path)
                ),
                FieldCondition(
                    key="chunk_index", match=MatchValue(value=citation.chunk_index)
                ),
            ]
        ),
        limit=1,
        with_payload=True,
    )
    if not points:
        return ""
    return points[0].payload.get("text", "")


def _run_question(
    question: EvalQuestion,
    *,
    settings: Settings,
    client: QdrantClient,
    cache: SemanticCache,
    embedding_cache: EmbeddingCache,
) -> EvalQuestionResult:
    """Answer `question` through the real pipeline (`answer_with_cache()`
    - the same function `POST /query` calls) and score the result against
    its ground truth.

    `retrieval_hit` is only computed for an `expected_answerable`
    question - it's `None` (not `False`) otherwise, since the metric
    doesn't apply and a forced `False` would incorrectly count against
    `retrieval_precision`'s denominator (see `report.py`).

    Faithfulness is only judged when the system produced an answer
    (`answered`) to a question that was actually `expected_answerable` -
    there's nothing to check a claim against when the answer is the
    canonical fallback, and when the question wasn't expected to be
    answerable at all, `answered` alone already fully determines
    `hallucinated` (below), so judging faithfulness on top would just be
    a wasted LLM round-trip confirming a verdict already known.
    `hallucinated` has two distinct triggers, not one: answering at all
    when `expected_answerable` is
    `False` (fabrication where nothing should have been said), or
    answering unfaithfully when it *was* expected to be answerable (a
    claim not actually supported by what was cited). A question that
    should have been answerable but got the fallback instead is neither -
    that's a retrieval miss, a different failure mode this function
    doesn't conflate with hallucination.
    """
    answer = answer_with_cache(
        question.query,
        question.user_tier,
        cache=cache,
        client=client,
        collection_name=settings.evaluation_qdrant_collection_name,
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

    # Substring containment, not exact equality - matches
    # answer_with_cache()'s own convention (`CANNOT_ANSWER_MESSAGE not in
    # answer.text`). Live-verified real gap: the generation model
    # sometimes wraps the fallback in surrounding whitespace (e.g. a
    # leading space), which `!=` fails to recognize as the fallback at
    # all, miscounting a correct "I don't know" as a fabricated answer.
    answered = CANNOT_ANSWER_MESSAGE not in answer.text
    cited_paths = [_normalize_path(c.relative_path) for c in answer.citations]

    retrieval_hit = None
    if question.expected_answerable:
        expected = {_normalize_path(p) for p in question.expected_source_paths}
        retrieval_hit = bool(expected & set(cited_paths))

    faithfulness = None
    if answered and question.expected_answerable:
        # Skipped when not expected_answerable, even though the system
        # did answer: hallucinated is already fully determined by
        # `answered` alone in that case (below) - a judge call would just
        # be a wasted LLM round-trip confirming a verdict already known.
        sources = "\n\n".join(
            f"[{citation.number}] "
            f"{_fetch_cited_source_text(client, settings.evaluation_qdrant_collection_name, citation)}"
            for citation in answer.citations
        )
        faithfulness = check_faithfulness(
            question.query,
            answer.text,
            sources,
            model=settings.evaluation_model,
            base_url=settings.ollama_base_url,
            timeout=settings.evaluation_timeout_seconds,
            temperature=settings.evaluation_temperature,
        )

    if not question.expected_answerable:
        hallucinated = answered
    else:
        hallucinated = answered and faithfulness is not None and not faithfulness.is_faithful

    return EvalQuestionResult(
        question_id=question.id,
        query=question.query,
        expected_answerable=question.expected_answerable,
        expected_source_paths=question.expected_source_paths,
        answer_text=answer.text,
        cited_paths=cited_paths,
        retrieval_hit=retrieval_hit,
        answered=answered,
        faithfulness=faithfulness,
        hallucinated=hallucinated,
    )


def run_evaluation(*, settings: Settings) -> EvaluationReport:
    """Run the full structured evaluation: index `settings.evaluation_corpus_path`
    into a dedicated Qdrant collection, answer every question in
    `settings.evaluation_questions_path` through the real pipeline, and
    return the aggregate report.

    Uses a *separate* Qdrant storage path/collection
    (`evaluation_qdrant_storage_path`/`evaluation_qdrant_collection_name`)
    from the app's real ones - an eval run must never mix eval-only
    fixture documents into, or read against, production data.

    Indexes via `run_sync_cycle()` (`ingestion/scheduler.py`) - the exact
    same ingestion/indexing code path the background sync job uses in
    production, not a separate eval-only reimplementation, so this
    measures the real system's actual behavior. Built via `settings.model_copy()`
    overriding only `watched_folder_path`/`qdrant_collection_name` -
    every other setting (chunk size, access tiers, embedding model, ...)
    stays identical to the real, configured system.

    A fresh `SemanticCache`/`EmbeddingCache` per run, matching this
    codebase's per-process cache scoping elsewhere - an eval run measures
    the system cold, not warmed by a previous run's answers.
    """
    questions = load_questions(settings.evaluation_questions_path)

    client = get_client(settings.evaluation_qdrant_storage_path)
    ensure_collection(
        client, settings.evaluation_qdrant_collection_name, settings.embedding_dimensions
    )

    eval_settings = settings.model_copy(
        update={
            "watched_folder_path": settings.evaluation_corpus_path,
            "qdrant_collection_name": settings.evaluation_qdrant_collection_name,
        }
    )
    run_sync_cycle(settings=eval_settings, client=client, previous_snapshot={})

    cache = SemanticCache()
    embedding_cache = EmbeddingCache()
    results = [
        _run_question(
            question,
            settings=eval_settings,
            client=client,
            cache=cache,
            embedding_cache=embedding_cache,
        )
        for question in questions
    ]

    return build_report(results)


def _report_path_for(results_dir: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return results_dir / f"eval-{timestamp}.json"


def main() -> None:
    """CLI entry point: `python -m agentic_rag.evaluation.runner`.

    Loads `Settings` from the environment/`.env`, same as the FastAPI app,
    so an eval run always measures the actually-configured system, not a
    separately-configured shadow one. Writes the full report as JSON to a
    timestamped file under `Settings.evaluation_results_path` (so repeat
    runs accumulate a history rather than overwriting each other) and
    prints the three summary metrics.
    """
    settings = Settings()
    report = run_evaluation(settings=settings)

    settings.evaluation_results_path.mkdir(parents=True, exist_ok=True)
    report_path = _report_path_for(settings.evaluation_results_path)
    report_path.write_text(json.dumps(report_to_json_dict(report), indent=2))

    print(f"retrieval_precision: {report.retrieval_precision}")
    print(f"faithfulness_rate:   {report.faithfulness_rate}")
    print(f"hallucination_rate:  {report.hallucination_rate}")
    print(f"report written to:   {report_path}")


if __name__ == "__main__":
    main()
