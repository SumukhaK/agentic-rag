from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.ingestion.scheduler import SyncCycleResult, run_sync_cycle
from agentic_rag.ingestion.snapshot_store import load_snapshot, save_snapshot
from agentic_rag.observability.loadtest_log import (
    configure_loadtest_logging,
    log_loadtest_batch,
    log_loadtest_run_complete,
)
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache

# A handful of fixed, representative queries against the generated
# corpus's own content (see loadtest/corpus_generator.py) - not the full
# eval/questions.json question set, which is scored for correctness
# against a small hand-written corpus. This phase measures one thing
# only: how long a real query takes to answer once the index actually
# holds the full target-scale document count, closing the gap the
# ingestion-only 150k theoretical analysis (README.md) left open.
_REPRESENTATIVE_QUERIES: list[tuple[str, str]] = [
    ("What was the score in the match reported as doc_00000?", "tier-1"),
    ("Which tactical approach was used in a tier-1 fixture?", "tier-1"),
    ("Summarize a match report from tier-2.", "tier-2"),
    ("What did the manager say after a tier-3 fixture?", "tier-3"),
]


@dataclass(frozen=True)
class LoadTestReport:
    """Summary of one full `run_load_test()` call: both the ingestion
    phase (batched indexing of the whole staged corpus) and the
    query-latency phase (a handful of real queries against the
    fully-loaded index)."""

    total_indexed: int
    total_ingestion_failures: int
    total_indexing_failures: int
    total_duration_seconds: float
    batch_count: int
    query_latencies_seconds: list[float]


def _next_batch(staged_dir: Path, watched_dir: Path, batch_size: int) -> list[Path]:
    """Which staged files still need to be copied into `watched_dir`, up
    to `batch_size` of them - a plain directory diff, not a separate
    progress-tracking file. This is what makes resumption after a crash
    free: files already copied into `watched_dir` (indexed or not) are
    simply excluded from "remaining work" here, and `run_sync_cycle()`'s
    own diff against the last-saved snapshot picks up anything present
    but not yet reflected there.
    """
    staged_files = sorted(p for p in staged_dir.rglob("*.md") if p.is_file())
    watched_relative = (
        {p.relative_to(watched_dir) for p in watched_dir.rglob("*.md") if p.is_file()}
        if watched_dir.exists()
        else set()
    )
    remaining = [
        path for path in staged_files if path.relative_to(staged_dir) not in watched_relative
    ]
    return remaining[:batch_size]


def _copy_batch(batch: list[Path], staged_dir: Path, watched_dir: Path) -> None:
    """Copy `batch` from `staged_dir` into `watched_dir`, preserving the
    tier-subfolder structure `sync_folder()`'s access-tier tagging
    depends on (a document's tier is its top-level subfolder under the
    watched folder)."""
    for path in batch:
        relative = path.relative_to(staged_dir)
        destination = watched_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def _index_one_batch(
    *,
    settings: Settings,
    client: QdrantClient,
    snapshot: dict,
    batch_index: int,
    run_start: float,
    total_indexed_so_far: int,
) -> tuple[dict, SyncCycleResult] | None:
    """Copy and index the next batch; `None` if nothing staged remains.

    `save_snapshot()` runs immediately after `run_sync_cycle()` returns -
    exactly what `run_sync_loop()` already does per cycle in production -
    so a crash loses at most one batch's worth of work, not the whole run.
    """
    batch = _next_batch(
        settings.loadtest_corpus_staging_path,
        settings.loadtest_watched_folder_path,
        settings.loadtest_batch_size,
    )
    if not batch:
        return None

    _copy_batch(
        batch, settings.loadtest_corpus_staging_path, settings.loadtest_watched_folder_path
    )
    batch_settings = settings.model_copy(
        update={
            "watched_folder_path": settings.loadtest_watched_folder_path,
            "qdrant_collection_name": settings.loadtest_qdrant_collection_name,
        }
    )

    cycle_start = time.monotonic()
    result, snapshot = run_sync_cycle(
        settings=batch_settings, client=client, previous_snapshot=snapshot
    )
    save_snapshot(settings.loadtest_sync_snapshot_path, snapshot)

    log_loadtest_batch(
        batch_index=batch_index,
        batch_size=len(batch),
        indexed_count=len(result.indexed),
        ingestion_failure_count=len(result.ingestion_failures),
        indexing_failure_count=len(result.indexing_failures),
        indexing_failure_paths=[f.relative_path for f in result.indexing_failures],
        duration_seconds=time.monotonic() - cycle_start,
        cumulative_indexed=total_indexed_so_far + len(result.indexed),
        cumulative_elapsed_seconds=time.monotonic() - run_start,
    )
    return snapshot, result


def _run_ingestion_phase(
    *, settings: Settings, client: QdrantClient
) -> tuple[int, int, int, int]:
    """Drip-feed the entire staged corpus into the watched folder in
    `loadtest_batch_size`-sized batches, one `run_sync_cycle()` call per
    batch, until nothing staged remains. Returns
    `(total_indexed, total_ingestion_failures, total_indexing_failures,
    batch_count)`.
    """
    snapshot = load_snapshot(settings.loadtest_sync_snapshot_path)
    run_start = time.monotonic()
    total_indexed = 0
    total_ingestion_failures = 0
    total_indexing_failures = 0
    batch_index = 0

    while True:
        outcome = _index_one_batch(
            settings=settings,
            client=client,
            snapshot=snapshot,
            batch_index=batch_index,
            run_start=run_start,
            total_indexed_so_far=total_indexed,
        )
        if outcome is None:
            break
        snapshot, result = outcome
        total_indexed += len(result.indexed)
        total_ingestion_failures += len(result.ingestion_failures)
        total_indexing_failures += len(result.indexing_failures)
        batch_index += 1

    return total_indexed, total_ingestion_failures, total_indexing_failures, batch_index


def _run_query_latency_phase(*, settings: Settings, client: QdrantClient) -> list[float]:
    """Answer `_REPRESENTATIVE_QUERIES` through `answer_with_cache()` -
    the same function `POST /query` calls, same pattern
    `evaluation/runner.py::_run_question()` already uses - against the
    now fully-loaded index, returning each query's wall-clock latency.
    A fresh `SemanticCache()` for this phase alone means every query is
    answered for real, not served from a cache hit against another one.
    """
    cache = SemanticCache()
    embedding_cache = EmbeddingCache()
    latencies: list[float] = []
    for query_text, tier in _REPRESENTATIVE_QUERIES:
        start = time.monotonic()
        answer_with_cache(
            query_text,
            tier,
            cache=cache,
            client=client,
            collection_name=settings.loadtest_qdrant_collection_name,
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
        latencies.append(time.monotonic() - start)
    return latencies


def run_load_test(*, settings: Settings) -> LoadTestReport:
    """Run the full Phase 8 load test against a dedicated Qdrant
    collection/storage path (never the app's real ones, matching
    `run_evaluation()`'s own isolation reasoning): batch-index the
    entire staged corpus (`_run_ingestion_phase`), then measure query
    latency against the fully-loaded index (`_run_query_latency_phase`).

    The Qdrant client is always closed before returning (`try`/`finally`)
    - embedded/local-mode Qdrant holds an exclusive file lock on its
    storage path for as long as the client stays open.
    """
    client = get_client(settings.loadtest_qdrant_storage_path)
    try:
        ensure_collection(
            client, settings.loadtest_qdrant_collection_name, settings.embedding_dimensions
        )
        run_start = time.monotonic()
        indexed, ingestion_failures, indexing_failures, batch_count = _run_ingestion_phase(
            settings=settings, client=client
        )
        query_latencies = _run_query_latency_phase(settings=settings, client=client)
        total_duration = time.monotonic() - run_start
    finally:
        client.close()

    return LoadTestReport(
        total_indexed=indexed,
        total_ingestion_failures=ingestion_failures,
        total_indexing_failures=indexing_failures,
        total_duration_seconds=total_duration,
        batch_count=batch_count,
        query_latencies_seconds=query_latencies,
    )


def _report_path_for(results_dir: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return results_dir / f"loadtest-{timestamp}.json"


def main() -> None:
    """CLI entry point: `python -m agentic_rag.loadtest.runner`.

    Loads `Settings` from the environment/`.env`, same as every other
    entry point in this codebase. Writes the full report as JSON to a
    timestamped file under `settings.loadtest_results_path` (repeat runs
    accumulate a history rather than overwriting each other), and emits
    one structured `loadtest_run_complete` log line via
    `observability/loadtest_log.py` alongside the per-batch lines already
    emitted during the run.
    """
    configure_loadtest_logging()
    settings = Settings()

    report = run_load_test(settings=settings)

    settings.loadtest_results_path.mkdir(parents=True, exist_ok=True)
    report_path = _report_path_for(settings.loadtest_results_path)
    report_path.write_text(json.dumps(asdict(report), indent=2))

    log_loadtest_run_complete(
        total_indexed=report.total_indexed,
        total_ingestion_failures=report.total_ingestion_failures,
        total_indexing_failures=report.total_indexing_failures,
        total_duration_seconds=report.total_duration_seconds,
        query_latencies_seconds=report.query_latencies_seconds,
        report_path=str(report_path),
    )

    print(f"total_indexed:         {report.total_indexed}")
    print(f"batch_count:           {report.batch_count}")
    print(f"total_duration_hours:  {report.total_duration_seconds / 3600:.2f}")
    print(f"query_latencies_s:     {report.query_latencies_seconds}")
    print(f"report written to:     {report_path}")


if __name__ == "__main__":
    main()
