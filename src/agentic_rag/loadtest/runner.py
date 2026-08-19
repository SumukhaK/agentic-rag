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
from agentic_rag.loadtest.corpus_generator import DEFAULT_ACCESS_TIERS
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
    ("What was the score in the match reported as doc_00000?", "employee"),
    ("Which tactical approach was used in a fixture from the employee tier?", "employee"),
    ("Summarize a match report from the manager tier.", "manager"),
    # "the club", not "the manager" - the manager tier is now a real access
    # tier value, and this query text would otherwise read as a coincidental
    # (and confusing) reference to it rather than a football team's manager.
    ("What did the club say after a fixture from the director tier?", "director"),
]


@dataclass(frozen=True)
class LoadTestReport:
    """Summary of one full `run_load_test()` call: both the ingestion
    phase (batched indexing of the whole staged corpus) and the
    query-latency phase (a handful of real queries against the
    fully-loaded index).

    `total_indexed` is a per-*invocation* count - documents this call
    actually indexed, which understates progress on a resumed run.
    `total_indexed_all_time` (derived from the persisted snapshot's
    length, not a per-invocation counter) is the true cumulative total
    across every invocation, indexed or not by this specific process -
    the number that actually answers "how far along is the whole
    corpus," which is what a report meant to be compared against the
    150k theoretical analysis needs.
    """

    total_indexed: int
    total_indexed_all_time: int
    total_ingestion_failures: int
    total_indexing_failures: int
    total_duration_seconds: float
    batch_count: int
    query_latencies_seconds: list[float]


def _next_batch(
    staged_dir: Path,
    watched_dir: Path,
    batch_size: int,
    *,
    staged_files: list[Path] | None = None,
) -> list[Path]:
    """Which staged files still need to be copied into `watched_dir`, up
    to `batch_size` of them - a plain directory diff, not a separate
    progress-tracking file. This is what makes resumption after a crash
    free: files already copied into `watched_dir` (indexed or not) are
    simply excluded from "remaining work" here, and `run_sync_cycle()`'s
    own diff against the last-saved snapshot picks up anything present
    but not yet reflected there.

    `staged_files`, if given, is a pre-listed and pre-sorted enumeration
    of `staged_dir` - the staged corpus never changes mid-run, so a
    caller invoking this once per batch across a long run (the batch
    loop in `_run_ingestion_phase`) should list `staged_dir` once and
    reuse it rather than re-walking and re-sorting the same, unchanging
    ~10,000-entry tree on every single call. Defaults to `None`, which
    re-derives it fresh - what a standalone/test caller gets automatically.
    """
    if staged_files is None:
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
    watched folder).

    Copies via a temp file + atomic rename (`Path.replace()`), matching
    `snapshot_store.save_snapshot()`'s own established pattern in this
    codebase - a plain `shutil.copyfile()` straight to `destination`
    would leave a half-written file in place if the process is killed
    mid-copy, and `_next_batch()`'s existence-only check would then treat
    that truncated file as "already copied" forever, since nothing else
    would ever overwrite it.
    """
    for path in batch:
        relative = path.relative_to(staged_dir)
        destination = watched_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(path, tmp_destination)
        tmp_destination.replace(destination)


def _index_one_batch(
    *,
    settings: Settings,
    client: QdrantClient,
    snapshot: dict,
    batch_index: int,
    run_start: float,
    total_indexed_so_far: int,
    staged_files: list[Path],
) -> tuple[dict, SyncCycleResult, bool]:
    """Copy the next batch (if any staged files remain uncopied) and run
    one indexing cycle regardless of whether anything new was copied
    this call.

    Always calling `run_sync_cycle()` - even on a call where nothing new
    was copied - is what catches a batch that was copied into
    `watched_dir` but never indexed, because a prior crash landed between
    `_copy_batch()` and `run_sync_cycle()`/`save_snapshot()`:
    `sync_folder()` diffs the *entire* current watched folder against the
    snapshot on every call, not just files copied this iteration, so a
    stranded batch gets picked up here even when `_next_batch()` sees
    nothing left to copy. Without this, a crash landing on the corpus's
    FINAL batch would leave it permanently un-indexed - `_next_batch()`
    would find nothing left to copy on restart and the loop would stop
    before ever re-running `run_sync_cycle()` for the stranded files.

    `access_tiers` is overridden to `DEFAULT_ACCESS_TIERS` (not left as
    whatever the main app's `Settings.access_tiers` is) - the load test's
    generated corpus always uses that fixed tier layout regardless of how
    the real app happens to be configured, so this keeps the two from
    silently disagreeing.

    Returns `(snapshot, result, more_work_possible)` - `more_work_possible`
    is true if a batch was copied this call or this cycle actually
    indexed/failed something (a sign there may still be stranded work to
    reconcile); the caller stops only once a cycle does neither.
    """
    batch = _next_batch(
        settings.loadtest_corpus_staging_path,
        settings.loadtest_watched_folder_path,
        settings.loadtest_batch_size,
        staged_files=staged_files,
    )
    if batch:
        _copy_batch(
            batch, settings.loadtest_corpus_staging_path, settings.loadtest_watched_folder_path
        )

    batch_settings = settings.model_copy(
        update={
            "watched_folder_path": settings.loadtest_watched_folder_path,
            "qdrant_collection_name": settings.loadtest_qdrant_collection_name,
            "access_tiers": list(DEFAULT_ACCESS_TIERS),
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
        ingestion_failure_paths=[f.relative_path for f in result.ingestion_failures],
        indexing_failure_paths=[f.relative_path for f in result.indexing_failures],
        duration_seconds=time.monotonic() - cycle_start,
        cumulative_indexed=total_indexed_so_far + len(result.indexed),
        cumulative_elapsed_seconds=time.monotonic() - run_start,
    )
    more_work_possible = bool(batch) or bool(result.indexed) or bool(
        result.ingestion_failures
    ) or bool(result.indexing_failures)
    return snapshot, result, more_work_possible


@dataclass(frozen=True)
class _IngestionPhaseResult:
    total_indexed: int
    total_indexed_all_time: int
    total_ingestion_failures: int
    total_indexing_failures: int
    batch_count: int


def _run_ingestion_phase(*, settings: Settings, client: QdrantClient) -> _IngestionPhaseResult:
    """Drip-feed the entire staged corpus into the watched folder in
    `loadtest_batch_size`-sized batches, one `run_sync_cycle()` call per
    batch, until a cycle neither copies anything new nor indexes/fails
    anything - see `_index_one_batch()`'s docstring for why the loop
    can't simply stop the moment nothing is left to *copy*.
    """
    snapshot = load_snapshot(settings.loadtest_sync_snapshot_path)
    run_start = time.monotonic()
    staged_files = sorted(
        p for p in settings.loadtest_corpus_staging_path.rglob("*.md") if p.is_file()
    )
    total_indexed = 0
    total_ingestion_failures = 0
    total_indexing_failures = 0
    batch_index = 0

    while True:
        snapshot, result, more_work_possible = _index_one_batch(
            settings=settings,
            client=client,
            snapshot=snapshot,
            batch_index=batch_index,
            run_start=run_start,
            total_indexed_so_far=total_indexed,
            staged_files=staged_files,
        )
        total_indexed += len(result.indexed)
        total_ingestion_failures += len(result.ingestion_failures)
        total_indexing_failures += len(result.indexing_failures)
        batch_index += 1
        if not more_work_possible:
            break

    return _IngestionPhaseResult(
        total_indexed=total_indexed,
        total_indexed_all_time=len(snapshot),
        total_ingestion_failures=total_ingestion_failures,
        total_indexing_failures=total_indexing_failures,
        batch_count=batch_index,
    )


def _run_query_latency_phase(*, settings: Settings, client: QdrantClient) -> list[float]:
    """Answer `_REPRESENTATIVE_QUERIES` through `answer_with_cache()` -
    the same function `POST /query` calls, same pattern
    `evaluation/runner.py::_run_question()` already uses - against the
    now fully-loaded index, returning each query's wall-clock latency.

    A fresh `SemanticCache()` *per query*, not shared across the loop -
    matches `_run_question()`'s own documented reasoning
    (`evaluation/runner.py`): two representative queries share a tier
    (`employee`), and a shared cache risks the second being served the
    first's cached answer if their embeddings land close enough, which
    would record a cache-lookup time instead of a real round trip.

    `known_tiers=DEFAULT_ACCESS_TIERS` (not `settings.access_tiers`) for
    the same isolation reason `_index_one_batch()`'s `access_tiers`
    override exists - these queries' hardcoded `employee`/`manager`/
    `director` `user_tier` values must validate against the load test's
    own fixed tier layout, not whatever the main app's
    `Settings.access_tiers` happens to be.
    """
    embedding_cache = EmbeddingCache()
    latencies: list[float] = []
    for query_text, tier in _REPRESENTATIVE_QUERIES:
        cache = SemanticCache()
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
            known_tiers=list(DEFAULT_ACCESS_TIERS),
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

    The query-latency phase only runs if the index actually holds
    anything (`total_indexed_all_time > 0`) - skipping it otherwise
    avoids recording real-looking-but-meaningless latencies (fast
    "cannot answer" fallbacks) against an empty collection, which would
    otherwise make a run that failed to index anything look like a
    completed, successful measurement.

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
        ingestion = _run_ingestion_phase(settings=settings, client=client)
        query_latencies = (
            _run_query_latency_phase(settings=settings, client=client)
            if ingestion.total_indexed_all_time > 0
            else []
        )
        total_duration = time.monotonic() - run_start
    finally:
        client.close()

    return LoadTestReport(
        total_indexed=ingestion.total_indexed,
        total_indexed_all_time=ingestion.total_indexed_all_time,
        total_ingestion_failures=ingestion.total_ingestion_failures,
        total_indexing_failures=ingestion.total_indexing_failures,
        total_duration_seconds=total_duration,
        batch_count=ingestion.batch_count,
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
    print(f"total_indexed_all_time:{report.total_indexed_all_time}")
    print(f"batch_count:           {report.batch_count}")
    print(f"total_duration_hours:  {report.total_duration_seconds / 3600:.2f}")
    print(f"query_latencies_s:     {report.query_latencies_seconds}")
    print(f"report written to:     {report_path}")


if __name__ == "__main__":
    main()
