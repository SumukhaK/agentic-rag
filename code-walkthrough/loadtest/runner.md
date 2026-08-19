# `loadtest/runner.py`

**Purpose:** This file drives the actual load test: it takes the synthetic corpus produced by `corpus_generator.py` and feeds it into the real ingestion/indexing pipeline in small, checkpointed (progress-saving) batches, rather than all at once — because indexing the full ~10,000-document corpus could take on the order of 30 hours, far too long to risk losing all progress if the process crashes partway through. After ingestion, it runs a handful of representative queries against the fully-loaded index to measure real query latency at scale. The whole thing runs against dedicated, isolated storage (its own Qdrant collection, its own watched folder, its own snapshot file) so it never touches the real application's data. It's designed to be safely interruptible and resumable: if the process is killed and restarted, it picks up roughly where it left off without needing a separate "progress" file, by comparing what's already on disk and what's already recorded in the index's own snapshot.

## Line-by-line walkthrough

### Lines 1-22 — Imports
```python
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
```
- `from __future__ import annotations` — makes type hints lazily evaluated as strings, allowing modern hint syntax across Python versions.
- `import json` — used to serialize the final load test report to a JSON file.
- `import shutil` — used for `shutil.copyfile()` when copying staged documents into the watched folder.
- `import time` — used for `time.monotonic()` timing measurements throughout (batch durations, total run duration, query latency).
- `from dataclasses import asdict, dataclass` — `dataclass` decorates the plain data-holding classes defined in this file (`LoadTestReport`, `_IngestionPhaseResult`); `asdict` converts a dataclass instance into a plain dictionary, used when serializing the report to JSON.
- `from datetime import datetime` — used to timestamp the report filename.
- `from pathlib import Path` — used throughout for filesystem paths.
- `from qdrant_client import QdrantClient` — the Qdrant vector database client type, used for type hints and passed through to ingestion/query functions.
- `from agentic_rag.config import Settings` — the application's central configuration object; the load test reads its own dedicated settings fields from this (e.g. `loadtest_batch_size`, `loadtest_qdrant_collection_name`).
- `from agentic_rag.embedding.cache import EmbeddingCache` — the cache that avoids re-computing embeddings for text already seen; used in the query-latency phase.
- `from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client` — `get_client` opens a connection to a (possibly local/embedded) Qdrant instance at a given storage path; `ensure_collection` makes sure the target collection exists with the right vector configuration before use.
- `from agentic_rag.ingestion.scheduler import SyncCycleResult, run_sync_cycle` — `run_sync_cycle` is the core pipeline function that scans a watched folder, ingests new/changed files, and indexes them; `SyncCycleResult` is the type describing what happened in one such cycle (what was indexed, what failed).
- `from agentic_rag.ingestion.snapshot_store import load_snapshot, save_snapshot` — functions to persist and reload the "snapshot" — the record of what's already been indexed — so `run_sync_cycle` knows what's new versus already processed, and so state survives a restart.
- `from agentic_rag.loadtest.corpus_generator import DEFAULT_ACCESS_TIERS` — imports the load test's fixed, dedicated tier list (see `corpus_generator.py`'s walkthrough) so this file's tier-dependent logic stays consistent with how the corpus was generated.
- `from agentic_rag.observability.loadtest_log import (...)` — structured logging helpers specific to the load test: `configure_loadtest_logging` sets up logging output, `log_loadtest_batch` logs progress after each batch, `log_loadtest_run_complete` logs a final summary line.
- `from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache` — `answer_with_cache` is the same function the real `/query` API endpoint uses to answer a question (checking a semantic cache first, then falling back to full retrieval); `SemanticCache` is the cache type it depends on. Reusing this exact function means the query-latency phase measures a realistic, production-representative code path.

### Lines 25-40 — `_REPRESENTATIVE_QUERIES`
```python
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
```
- The comment explains the purpose and scope of this list: unlike the project's full evaluation question set (`eval/questions.json`, which checks answer *correctness* against a small hand-crafted corpus), this list exists purely to measure query *latency* — how long a real answer takes once the index is holding the full, large-scale document count. This closes a gap left by an earlier, purely theoretical estimate (in the README) of what indexing 150,000 documents would look like, by actually measuring query performance against a realistically large index.
- `_REPRESENTATIVE_QUERIES: list[tuple[str, str]] = [...]` — defines a fixed list of `(query_text, access_tier)` pairs. Each tuple pairs a natural-language question with the access tier the querying user is simulated to have.
  - The first two queries use the `employee` tier, one asking about a specific document (`doc_00000`) and one asking a more general tactical question.
  - The third asks for a summary from the `manager` tier.
  - The fourth asks about the `director` tier, with an inline comment explaining a subtle wording choice: the query deliberately says "the club" instead of "the manager" to avoid ambiguity, since "manager" is now also a real, distinct access-tier name in this system — using the word "manager" in the query text could be misread as referring to that tier rather than a football team's on-pitch manager.

### Lines 43-66 — `LoadTestReport` dataclass
```python
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
```
- `@dataclass(frozen=True)` — marks this class as an immutable data container (`frozen=True` means its fields can't be reassigned after creation, preventing accidental mutation of a report once built); `dataclass` auto-generates the constructor, equality, and representation methods from the field list below.
- The docstring explains a subtle but important distinction between two of the fields: `total_indexed` only counts documents indexed *during this specific call* to `run_load_test()`, so if the load test is stopped and resumed, this number will look artificially low on the resumed run (it doesn't include work done by earlier, separate invocations). `total_indexed_all_time`, by contrast, is read from the persisted snapshot itself (the durable record of everything ever indexed), so it reflects true cumulative progress across every run — the number actually meaningful when comparing against the project's 150,000-document theoretical scaling analysis.
- `total_indexed: int` — documents indexed by this specific call.
- `total_indexed_all_time: int` — true cumulative total ever indexed, from the snapshot.
- `total_ingestion_failures: int` — count of documents that failed during the ingestion (reading/parsing) step.
- `total_indexing_failures: int` — count of documents that failed during the indexing (embedding/storing) step.
- `total_duration_seconds: float` — how long this entire `run_load_test()` call took, wall-clock.
- `batch_count: int` — how many batches were processed during the ingestion phase.
- `query_latencies_seconds: list[float]` — one latency measurement per representative query answered in the query-latency phase.

### Lines 69-102 — `_next_batch`
```python
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
```
- The docstring explains the core resumability design: instead of maintaining a separate "how far did I get" progress file that could itself get out of sync after a crash, this function simply compares what's on disk in the staging directory against what's already been copied into the watched directory. Anything already present in the watched folder — whether or not it's actually been indexed yet — is considered "not remaining work" here. The genuinely-not-yet-indexed-but-already-copied case is instead handled by `run_sync_cycle()`'s own comparison against the last saved snapshot (the record of what's been indexed), so between this diff and that one, nothing needed to track progress ever needs its own dedicated state file.
- It also explains the `staged_files` parameter's purpose: since the staged corpus is a fixed, unchanging set of ~10,000 files once generated, re-scanning and re-sorting the entire directory tree on every single batch call (as the loop in `_run_ingestion_phase` does) would be wasteful. Callers that already have a sorted listing can pass it in and reuse it; the default of `None` causes the function to compute it fresh, which is convenient for standalone use or tests that call this function in isolation.
- `def _next_batch(staged_dir, watched_dir, batch_size, *, staged_files=None) -> list[Path]:` — the function signature: takes the staging directory, the watched directory, how many files to return at most, and an optional pre-computed file listing (keyword-only, after the bare `*`).
- `if staged_files is None: staged_files = sorted(p for p in staged_dir.rglob("*.md") if p.is_file())` — if no pre-listed files were provided, recursively finds every `.md` file under the staging directory (filtering out any non-file matches, like directories that might match the glob), and sorts them for a stable, deterministic order.
- `watched_relative = ({...} if watched_dir.exists() else set())` — builds a set of all `.md` files already present in the watched directory, expressed as paths *relative to* `watched_dir` (so they can be compared against staged files' relative paths regardless of the two directories' absolute locations). If the watched directory doesn't exist yet (e.g. very first run), this is an empty set instead of raising an error.
- `remaining = [path for path in staged_files if path.relative_to(staged_dir) not in watched_relative]` — filters the staged files down to only those whose relative path isn't already present in the watched folder — i.e., files not yet copied over.
- `return remaining[:batch_size]` — returns at most `batch_size` of the remaining files (a Python slice naturally handles the case where fewer than `batch_size` remain, returning all of them).

### Lines 105-125 — `_copy_batch`
```python
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
```
- The docstring explains two things. First, why relative paths are preserved when copying: the ingestion pipeline (`sync_folder()`, elsewhere in the codebase) determines a document's access tier from its top-level subfolder name under the watched directory, so copying must keep each document nested the same way it was staged (e.g. `employee/doc_00042.md` stays `employee/doc_00042.md`). Second, why the copy uses a temp-file-plus-rename pattern rather than a direct copy: if the process were killed partway through a plain `shutil.copyfile()` straight to the final destination, a half-written, truncated file could be left behind at that final path. Because `_next_batch()` only checks *existence* (not completeness) when deciding what's "already copied," that truncated file would then be permanently treated as done and never retried or overwritten — silently corrupting that document's ingestion forever. Writing to a `.tmp` path first and only renaming it to the real destination once the copy is fully complete avoids this: an interrupted copy leaves behind an orphaned `.tmp` file, not a corrupted real one, and `Path.replace()` is atomic (the rename either fully happens or doesn't happen at all, with no in-between state visible to other processes/restarts). The docstring notes this mirrors an existing pattern already used elsewhere in the codebase (`snapshot_store.save_snapshot()`).
- `for path in batch:` — iterates over each file selected by `_next_batch()`.
  - `relative = path.relative_to(staged_dir)` — computes this file's path relative to the staging root (e.g. `employee/doc_00042.md`), stripping the absolute staging directory prefix.
  - `destination = watched_dir / relative` — builds the corresponding destination path under the watched directory, preserving the same relative structure (and therefore the same tier subfolder).
  - `destination.parent.mkdir(parents=True, exist_ok=True)` — ensures the destination's parent directory (the tier subfolder under the watched directory) exists, creating it and any missing parents if necessary.
  - `tmp_destination = destination.with_suffix(destination.suffix + ".tmp")` — computes a temporary filename by appending `.tmp` to the existing file extension (e.g. `doc_00042.md` becomes `doc_00042.md.tmp`).
  - `shutil.copyfile(path, tmp_destination)` — copies the file's contents from the staging location to the temporary destination.
  - `tmp_destination.replace(destination)` — atomically renames the temp file to its final destination name, making the fully-copied file appear at its real path all at once.

### Lines 128-205 — `_index_one_batch`
```python
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
```
- The docstring lays out three important design decisions, each addressing a specific failure scenario:
  1. `run_sync_cycle()` (the actual indexing step) is always called, even when there was nothing new to copy this time. This matters because `run_sync_cycle`'s underlying `sync_folder()` logic compares the *entire* current watched folder against the last saved snapshot every time it runs — not just files copied in this particular call. So if a previous run crashed at exactly the wrong moment (after files were copied into the watched folder, but before they were indexed and the snapshot saved), those "stranded" files would otherwise never get indexed: on the next run, `_next_batch()` would correctly see they're already copied and report nothing left to copy, and if indexing were only triggered when there was something new to copy, the loop would stop right there, permanently leaving those files un-indexed. Calling `run_sync_cycle()` unconditionally on every iteration ensures such stranded work always gets picked up and reconciled.
  2. The docstring explicitly calls out the worst case this protects against: a crash landing exactly on the corpus's *final* batch, which — without this safeguard — would leave that last batch permanently un-indexed, since there'd be nothing left to copy on restart to trigger another cycle.
  3. `access_tiers` is deliberately overridden to the load test's own fixed `DEFAULT_ACCESS_TIERS`, not whatever the real application's `Settings.access_tiers` happens to be configured as — for the same self-containment reason discussed in `corpus_generator.py`.
  - The docstring also documents the return value's meaning: `more_work_possible` signals whether the caller's loop should keep going — true if either a batch was copied this call, or the indexing cycle actually indexed or failed something (both signs that there might still be stranded work to reconcile in a future iteration). The loop only stops once a cycle does neither.
- `def _index_one_batch(*, settings, client, snapshot, batch_index, run_start, total_indexed_so_far, staged_files) -> tuple[dict, SyncCycleResult, bool]:` — all parameters are keyword-only (the bare `*`), which forces every call site to name its arguments explicitly, improving readability and preventing mistakes when so many parameters share similar-sounding names.
- `batch = _next_batch(settings.loadtest_corpus_staging_path, settings.loadtest_watched_folder_path, settings.loadtest_batch_size, staged_files=staged_files)` — determines the next chunk of staged files (up to the configured batch size) still needing to be copied.
- `if batch: _copy_batch(batch, settings.loadtest_corpus_staging_path, settings.loadtest_watched_folder_path)` — if there's anything to copy, physically copies those files into the watched folder (skipping this call entirely when `batch` is empty, since there's nothing to copy).
- `batch_settings = settings.model_copy(update={...})` — creates a modified copy of the application settings (using Pydantic's `model_copy`, which doesn't mutate the original `settings` object) with three fields overridden for this indexing cycle: the watched folder path (pointed at the load test's dedicated watched folder, not the real app's), the Qdrant collection name (pointed at the load test's dedicated collection), and the access tiers list (forced to the load test's fixed tier list, converted from the `DEFAULT_ACCESS_TIERS` tuple to a list since that's presumably the type `Settings.access_tiers` expects).
- `cycle_start = time.monotonic()` — records the start time of this indexing cycle for later duration measurement (`time.monotonic()` is used rather than wall-clock time because it's immune to system clock adjustments, making it suitable for measuring elapsed durations).
- `result, snapshot = run_sync_cycle(settings=batch_settings, client=client, previous_snapshot=snapshot)` — runs one actual ingestion/indexing cycle using the overridden settings, the shared Qdrant client, and the previous snapshot (state from prior batches); returns both the outcome of this cycle (`result`) and the updated snapshot reflecting anything newly indexed.
- `save_snapshot(settings.loadtest_sync_snapshot_path, snapshot)` — persists the updated snapshot to disk immediately after each cycle, so that if the process crashes right after this line, the next run will correctly know what's already been indexed.
- `log_loadtest_batch(...)` — emits a structured log line summarizing this batch: which batch number it was, how many files were in it, how many were successfully indexed, how many failed at the ingestion stage versus the indexing stage (with the specific relative file paths of each failure), how long this cycle took, and two running totals — cumulative documents indexed so far in this run, and cumulative elapsed time since the run started.
- `more_work_possible = bool(batch) or bool(result.indexed) or bool(result.ingestion_failures) or bool(result.indexing_failures)` — computes the signal the docstring described: true if a batch was copied this call, or if the cycle indexed anything, or if it recorded any ingestion or indexing failures — any of these suggest there could be more work to do or reconcile on a subsequent call.
- `return snapshot, result, more_work_possible` — returns the updated snapshot (to feed into the next call), the cycle's result (for the caller to accumulate totals from), and the continue/stop signal.

### Lines 208-214 — `_IngestionPhaseResult` dataclass
```python
@dataclass(frozen=True)
class _IngestionPhaseResult:
    total_indexed: int
    total_indexed_all_time: int
    total_ingestion_failures: int
    total_indexing_failures: int
    batch_count: int
```
- `@dataclass(frozen=True)` — same rationale as `LoadTestReport`: an immutable, auto-generated data container.
- This private (leading underscore) dataclass is an internal summary specifically for the ingestion phase alone, before the query-latency phase's results are folded in to produce the final public `LoadTestReport`. Its five fields mirror the correspondingly-named fields on `LoadTestReport`.

### Lines 217-257 — `_run_ingestion_phase`
```python
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
```
- The docstring summarizes the phase's job: repeatedly call `_index_one_batch()` — each call copying one more batch of staged files and running an indexing cycle — until a cycle signals there's genuinely nothing more to do (neither new files to copy nor anything indexed/failed this time). It refers back to `_index_one_batch()`'s docstring for the detailed reasoning on why the loop can't simply stop as soon as nothing is left to copy (the "stranded final batch" crash scenario explained above).
- `snapshot = load_snapshot(settings.loadtest_sync_snapshot_path)` — loads whatever indexing progress snapshot already exists on disk (an empty/fresh one if this is the very first run), which is what makes resuming after a restart possible.
- `run_start = time.monotonic()` — records when this ingestion phase began, used to compute cumulative elapsed time in log messages.
- `staged_files = sorted(p for p in settings.loadtest_corpus_staging_path.rglob("*.md") if p.is_file())` — lists and sorts every staged markdown document once, up front — this is the pre-computed listing that gets passed into every `_index_one_batch()` / `_next_batch()` call for the rest of the run, avoiding the cost of re-walking the ~10,000-file directory tree on every single batch (as explained in `_next_batch()`'s docstring).
- `total_indexed = 0`, `total_ingestion_failures = 0`, `total_indexing_failures = 0`, `batch_index = 0` — running totals and a batch counter, all initialized before the loop begins.
- `while True:` — an unbounded loop that only terminates via the explicit `break` below (used because the natural stopping condition — "no more work possible" — is only known after each iteration runs, not in advance).
  - `snapshot, result, more_work_possible = _index_one_batch(...)` — processes one more batch, passing along the current snapshot, current totals (for logging), and the pre-listed staged files.
  - `total_indexed += len(result.indexed)` — accumulates the count of documents indexed in this batch.
  - `total_ingestion_failures += len(result.ingestion_failures)` — accumulates ingestion failure count.
  - `total_indexing_failures += len(result.indexing_failures)` — accumulates indexing failure count.
  - `batch_index += 1` — advances the batch counter for the next iteration (and for the final `batch_count`).
  - `if not more_work_possible: break` — exits the loop once a cycle indicates there's truly nothing left to do.
- The final `return _IngestionPhaseResult(...)` builds the phase's summary: `total_indexed` and the two failure counts are this call's accumulated totals; `total_indexed_all_time=len(snapshot)` derives the true cumulative total from the size of the final snapshot (the snapshot presumably records one entry per successfully indexed document across all runs ever, not just this one); `batch_count=batch_index` is the number of batches processed.

### Lines 260-296 — `_run_query_latency_phase`
```python
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
            embedding_cache=embedding_cache,
            known_tiers=list(DEFAULT_ACCESS_TIERS),
            settings=settings,
        )
        latencies.append(time.monotonic() - start)
    return latencies
```
- The docstring explains three design decisions:
  1. This phase deliberately reuses `answer_with_cache()` — the exact same function the real `POST /query` API endpoint calls — so the measured latency reflects a genuinely realistic query path, not a simplified stand-in. It notes this mirrors a pattern already established elsewhere in the codebase (`evaluation/runner.py`'s `_run_question()`).
  2. A brand-new `SemanticCache()` instance is created for *each* query in the loop, rather than sharing one cache across all queries. The reasoning: two of the representative queries share the same access tier (`employee`), and if they shared a semantic cache, the second query's embedding might land close enough to the first's to be served a cached answer instead of doing a real retrieval — which would measure a fast cache hit rather than the genuine end-to-end latency the phase is trying to capture.
  3. `known_tiers` is passed as the load test's own fixed `DEFAULT_ACCESS_TIERS`, not the main application's configurable `Settings.access_tiers` — for the same reason as elsewhere in this file: the representative queries' hardcoded tier values (`employee`, `manager`, `director`) need to validate against the load test's fixed tier layout, independent of how the real app happens to be configured.
- `embedding_cache = EmbeddingCache()` — creates one embedding cache shared across all queries in this phase (unlike the semantic cache, this one is intentionally shared — the docstring's isolation concern is specific to the semantic *answer* cache, not the embedding cache).
- `latencies: list[float] = []` — accumulator for each query's measured latency.
- `for query_text, tier in _REPRESENTATIVE_QUERIES:` — iterates over each fixed representative query and its associated access tier.
  - `cache = SemanticCache()` — creates a fresh, empty semantic cache for this query only, per the reasoning above.
  - `start = time.monotonic()` — marks the start of this query's timing window.
  - `answer_with_cache(query_text, tier, cache=cache, client=client, collection_name=settings.loadtest_qdrant_collection_name, embedding_cache=embedding_cache, known_tiers=list(DEFAULT_ACCESS_TIERS), settings=settings)` — actually answers the query end-to-end, using the load test's dedicated Qdrant collection, the per-query fresh semantic cache, the shared embedding cache, the load test's fixed tier list for validation, and the application settings. The return value (the actual answer) is discarded — only timing matters here.
  - `latencies.append(time.monotonic() - start)` — records how long that call took and adds it to the results list.
- `return latencies` — returns the list of per-query latencies, one float per representative query, in the same order they were defined.

### Lines 299-341 — `run_load_test`
```python
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
```
- This is the top-level orchestration function tying both phases together. The docstring explains three points:
  1. Everything runs against a dedicated Qdrant collection and storage path — never the real application's — mirroring an isolation approach already used by the project's evaluation runner (`run_evaluation()`), so a load test run can never corrupt or pollute production/real data.
  2. The query-latency phase is skipped entirely if the ingestion phase didn't actually index anything (`total_indexed_all_time > 0` check). The reasoning: querying a genuinely empty index would still return quickly (with a fast "cannot answer" style fallback response), and recording that fast response as a "latency measurement" would be misleading — it would make a load test run that completely failed to index anything look, superficially, like a successful, fast, completed measurement.
  3. The Qdrant client is always closed in a `finally` block, guaranteeing cleanup even if an exception occurs during either phase. This matters specifically because Qdrant running in embedded/local mode holds an exclusive lock on its storage directory for as long as the client connection stays open — failing to close it would leave that storage path locked and unusable by any subsequent process (including a later invocation of this same script).
- `client = get_client(settings.loadtest_qdrant_storage_path)` — opens a Qdrant client connected to the load test's dedicated storage path.
- `try:` — begins the block whose cleanup (closing the client) is guaranteed via `finally` below.
  - `ensure_collection(client, settings.loadtest_qdrant_collection_name, settings.embedding_dimensions)` — makes sure the load test's dedicated collection exists in Qdrant, created with the correct embedding vector dimensionality if it doesn't already exist.
  - `run_start = time.monotonic()` — marks the start of the full run for total duration measurement.
  - `ingestion = _run_ingestion_phase(settings=settings, client=client)` — runs the entire batched ingestion phase, returning its summary.
  - `query_latencies = (_run_query_latency_phase(...) if ingestion.total_indexed_all_time > 0 else [])` — conditionally runs the query-latency phase only if the index actually holds documents, per the reasoning above; otherwise uses an empty list.
  - `total_duration = time.monotonic() - run_start` — computes the total wall-clock duration of both phases combined.
- `finally: client.close()` — closes the Qdrant client unconditionally, whether the `try` block succeeded or raised an exception, releasing the storage lock.
- The final `return LoadTestReport(...)` — assembles and returns the public report dataclass, pulling `total_indexed`, `total_indexed_all_time`, and both failure counts from the ingestion phase's summary, `total_duration_seconds` from the measured total duration, `batch_count` from the ingestion summary, and `query_latencies_seconds` from the query phase's result (or the empty list if skipped).

### Lines 344-346 — `_report_path_for`
```python
def _report_path_for(results_dir: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return results_dir / f"loadtest-{timestamp}.json"
```
- `def _report_path_for(results_dir, now=None) -> Path:` — a small helper that computes where to write a report file; accepts an optional `now` timestamp (defaulting to `None`) primarily so tests can supply a fixed, predictable time instead of relying on the real current time.
- `timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")` — uses the provided `now` if given, otherwise the actual current time, formatted as a compact sortable timestamp string (e.g. `20260819T143000`).
- `return results_dir / f"loadtest-{timestamp}.json"` — builds the final report file path by combining the results directory with a filename that embeds the timestamp — ensuring each run produces a uniquely named file rather than overwriting a previous run's report.

### Lines 349-383 — `main` (CLI entry point)
```python
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
```
- The docstring notes this is meant to be run as `python -m agentic_rag.loadtest.runner`, that `Settings` is loaded the same way every other entry point in the codebase loads configuration (from environment variables / a `.env` file), and that report files accumulate in a results directory over time rather than overwriting each other on repeat runs — building up a history of load test results. It also notes a structured "run complete" log line is emitted at the end, complementing the per-batch log lines already emitted during the run by `_index_one_batch()`.
- `configure_loadtest_logging()` — sets up the logging configuration specific to load test runs before anything else happens.
- `settings = Settings()` — loads the application configuration (including all the `loadtest_*` fields) from the environment/`.env` file.
- `report = run_load_test(settings=settings)` — runs the entire load test (both phases) and captures the resulting report.
- `settings.loadtest_results_path.mkdir(parents=True, exist_ok=True)` — ensures the results directory exists before trying to write into it.
- `report_path = _report_path_for(settings.loadtest_results_path)` — computes this run's uniquely timestamped report file path.
- `report_path.write_text(json.dumps(asdict(report), indent=2))` — converts the `LoadTestReport` dataclass instance into a plain dictionary (`asdict`), serializes it as nicely indented JSON, and writes it to the report file.
- `log_loadtest_run_complete(...)` — emits the final structured summary log line, passing through the report's key figures (total indexed, both failure counts, total duration, query latencies) plus the path where the full JSON report was written.
- The five `print(...)` statements — output a concise, human-readable summary directly to the console/terminal for whoever ran the command: total documents indexed this run, the true all-time cumulative total, how many batches were processed, the total duration expressed in hours (converted from seconds and formatted to two decimal places for readability, since a load test run can span many hours), the list of per-query latencies, and finally where the full JSON report file was written.

### Lines 386-387 — Script entry guard
```python
if __name__ == "__main__":
    main()
```
- Standard Python idiom ensuring `main()` only runs when this file is executed directly as a script, not when it's imported as a module by other code.
