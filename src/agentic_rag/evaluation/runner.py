from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient

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
from agentic_rag.indexing.upsert import _point_id
from agentic_rag.ingestion.scheduler import run_sync_cycle
from agentic_rag.orchestration.answer import Citation
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache


class EvaluationIndexingError(Exception):
    """Raised when indexing `eval/corpus/` itself fails - a corpus setup
    problem, not a pipeline measurement. Every question's score depends on
    the corpus having actually been indexed correctly; silently proceeding
    with a partially-indexed collection would produce a `retrieval_precision`
    indistinguishable from a real regression, for the wrong reason."""


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
    retrieval result - so this fetches the payload `index_document()`
    (`indexing/upsert.py`) already stores there. Looked up via
    `_point_id(relative_path, chunk_index)` - the same deterministic ID
    every point was written under - and a single `client.retrieve()` by
    ID, rather than a filtered `scroll()`: every point this function
    looks for already has a known ID, so re-deriving it via a second,
    differently-shaped filter-search mechanism would be both slower and a
    second place the `(relative_path, chunk_index)` → point mapping has
    to stay in sync with `indexing/upsert.py`'s own.

    Returns `""` if the point can't be found (a stale citation against a
    reindexed corpus) rather than raising - the faithfulness judge will
    then correctly see nothing to support the claim.
    """
    points = client.retrieve(
        collection_name=collection_name,
        ids=[_point_id(citation.relative_path, citation.chunk_index)],
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
    embedding_cache: EmbeddingCache,
) -> EvalQuestionResult:
    """Answer `question` through the real pipeline (`answer_with_cache()`
    - the same function `POST /query` calls) and score the result against
    its ground truth.

    A fresh `SemanticCache()` is created for *this question alone*, not
    shared across the run - `SemanticCache` serves a hit whenever a
    query's embedding is similar enough (by default `>=0.95` cosine) to
    an earlier *same-tier* cached query, returning that earlier answer
    as-is with no error. Sharing one cache across every question in a run
    risks a later question being silently scored against an earlier,
    unrelated question's answer if their embeddings happen to land close
    enough - a real risk as an eval question set grows to include
    paraphrased/near-duplicate questions, which is a natural way such
    sets expand over time. `embedding_cache` doesn't have this risk (it's
    keyed by exact text, not fuzzy similarity, so sharing it across
    questions only ever avoids recomputing an identical embedding) and
    stays shared for that legitimate reuse.

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
    when `expected_answerable` is `False` (fabrication where nothing
    should have been said), or answering unfaithfully when it *was*
    expected to be answerable (a claim not actually supported by what was
    cited). A question that should have been answerable but got the
    fallback instead is neither - that's a retrieval miss, a different
    failure mode this function doesn't conflate with hallucination.

    Any exception raised anywhere in this process (a judge-model
    timeout/OOM - this session's own hardware has a documented history of
    exactly that, an Ollama connection failure, ...) is caught and
    recorded in the returned result's `error` field rather than
    propagating - one question's infrastructure failure must not discard
    every other, already-computed result in the same run, the same
    per-item isolation `run_sync_cycle()` already applies one layer down
    for indexing.

    `duration_seconds` is measured from entry to either return path (via
    `time.monotonic()`, immune to wall-clock adjustments) - it covers the
    whole round trip (retrieval, generation, and the faithfulness judge
    call when it runs), including a question that ultimately errored, so
    "how long before it failed" is captured too.
    """
    start = time.monotonic()
    try:
        cache = SemanticCache()
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
        # answer_with_cache()'s own convention (`CANNOT_ANSWER_MESSAGE not
        # in answer.text`). Live-verified real gap: the generation model
        # sometimes wraps the fallback in surrounding whitespace (e.g. a
        # leading space), which `!=` fails to recognize as the fallback
        # at all, miscounting a correct "I don't know" as a fabricated
        # answer.
        answered = CANNOT_ANSWER_MESSAGE not in answer.text
        cited_paths = [_normalize_path(c.relative_path) for c in answer.citations]

        retrieval_hit = None
        if question.expected_answerable:
            expected = {_normalize_path(p) for p in question.expected_source_paths}
            retrieval_hit = bool(expected & set(cited_paths))

        faithfulness = None
        if answered and question.expected_answerable:
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
            hallucinated = (
                answered and faithfulness is not None and not faithfulness.is_faithful
            )

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
            error=None,
            duration_seconds=time.monotonic() - start,
        )
    except Exception as exc:  # noqa: BLE001 - isolate one bad question, see docstring
        return EvalQuestionResult(
            question_id=question.id,
            query=question.query,
            expected_answerable=question.expected_answerable,
            expected_source_paths=question.expected_source_paths,
            answer_text="",
            cited_paths=[],
            retrieval_hit=None,
            answered=False,
            faithfulness=None,
            hallucinated=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - start,
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

    **The eval collection is deleted and recreated fresh at the start of
    every run**, not reused across runs. `run_sync_cycle()`'s
    incremental-diff design is built for a long-running background loop
    that persists its snapshot between cycles (`ingestion/scheduler.py`);
    this function always starts from `previous_snapshot={}`, which -
    against a *persisted* collection - would mean a corpus file removed
    or renamed between eval runs leaves its old chunks stranded in the
    index forever (`diff_snapshots({}, current)` can never report
    anything as deleted). Recreating the collection first makes
    `previous_snapshot={}` correct instead of merely convenient: every
    run starts from a guaranteed-empty collection, so "everything looks
    new" is always true, and there's nothing stale left over to strand.

    Indexes via `run_sync_cycle()` (`ingestion/scheduler.py`) - the exact
    same ingestion/indexing code path the background sync job uses in
    production, not a separate eval-only reimplementation, so this
    measures the real system's actual behavior. Its returned
    `SyncCycleResult` is checked, not discarded: any ingestion/indexing/
    deletion failure while setting up the corpus raises
    `EvaluationIndexingError` rather than silently proceeding to score
    every question against a partially-indexed collection.

    `eval_settings` (via `settings.model_copy()`, overriding only
    `watched_folder_path`/`qdrant_collection_name`) is used *only* for
    the `run_sync_cycle()` indexing call, where the override actually
    matters - `_run_question()` reads `evaluation_qdrant_collection_name`
    directly from the plain `settings` object it's already passed, so
    threading the copy through it too would add indirection with no
    behavioral effect.

    The Qdrant client is always closed before returning or raising
    (`try`/`finally`) - embedded/local-mode Qdrant holds an exclusive
    file lock on its storage path for as long as the client stays open,
    the same reasoning `api/app.py`'s `lifespan` already documents for
    the production client.
    """
    questions = load_questions(settings.evaluation_questions_path)

    client = get_client(settings.evaluation_qdrant_storage_path)
    try:
        if client.collection_exists(settings.evaluation_qdrant_collection_name):
            client.delete_collection(settings.evaluation_qdrant_collection_name)
        ensure_collection(
            client, settings.evaluation_qdrant_collection_name, settings.embedding_dimensions
        )

        eval_settings = settings.model_copy(
            update={
                "watched_folder_path": settings.evaluation_corpus_path,
                "qdrant_collection_name": settings.evaluation_qdrant_collection_name,
            }
        )
        sync_result, _snapshot = run_sync_cycle(
            settings=eval_settings, client=client, previous_snapshot={}
        )
        if (
            sync_result.ingestion_failures
            or sync_result.indexing_failures
            or sync_result.deletion_failures
        ):
            raise EvaluationIndexingError(
                "failed to index the eval corpus: "
                f"{len(sync_result.ingestion_failures)} ingestion failure(s), "
                f"{len(sync_result.indexing_failures)} indexing failure(s), "
                f"{len(sync_result.deletion_failures)} deletion failure(s) - "
                f"{[f.relative_path for f in sync_result.ingestion_failures]}"
            )

        embedding_cache = EmbeddingCache()
        results = [
            _run_question(
                question,
                settings=settings,
                client=client,
                embedding_cache=embedding_cache,
            )
            for question in questions
        ]
    finally:
        client.close()

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
    prints the summary metrics.
    """
    settings = Settings()
    report = run_evaluation(settings=settings)

    settings.evaluation_results_path.mkdir(parents=True, exist_ok=True)
    report_path = _report_path_for(settings.evaluation_results_path)
    report_path.write_text(json.dumps(report_to_json_dict(report), indent=2))

    print(f"retrieval_precision: {report.retrieval_precision}")
    print(f"faithfulness_rate:   {report.faithfulness_rate}")
    print(f"hallucination_rate:  {report.hallucination_rate}")
    print(f"errored_count:       {report.errored_count}")
    print(f"avg_duration_seconds:{report.average_duration_seconds}")
    print(f"report written to:   {report_path}")


if __name__ == "__main__":
    main()
