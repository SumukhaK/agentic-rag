# `ingestion/scheduler.py`

**Purpose:** This is the file that turns ingestion from "a function you can call" into "a background process that keeps the search index continuously in sync with the watched folder, forever, without blocking the rest of the application." It has two halves. `run_sync_cycle()` does the work for a *single* pass: it detects what changed on disk, pushes those changes into Qdrant (the vector database), and figures out what the next cycle should treat as "previously seen" — carefully making sure that anything which failed gets retried next time rather than silently forgotten. `run_sync_loop()` wraps that single pass in an infinite loop that runs forever inside the same process as the FastAPI web server, sleeping between cycles, persisting progress, logging outcomes, and handling graceful shutdown when the application is stopped. This file is dense with defensive reasoning in its docstrings because it sits at the intersection of several genuinely hard problems: partial failure (a document or deletion can fail independently of others), concurrency (it must not block the web server's request handling), and safe cancellation (it must not race with the app shutting down and closing the database connection out from under it).

## Line-by-line walkthrough

### Lines 1-17 — Imports
```python
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.indexing.upsert import delete_document, index_document
from agentic_rag.ingestion.pipeline import IngestionFailure
from agentic_rag.ingestion.snapshot_store import save_snapshot
from agentic_rag.ingestion.sync import sync_folder
from agentic_rag.ingestion.watcher import FileState
from agentic_rag.observability.sync_log import log_sync_cycle, log_sync_cycle_error
```
- `from __future__ import annotations` — defers type hint evaluation for modern syntax.
- `import asyncio` — Python's async/concurrency library, needed to run the sync loop as a coroutine that cooperates with the rest of the FastAPI app's event loop, and to run blocking work in a background thread (`asyncio.to_thread`) without freezing that event loop.
- `import threading` — used for a `threading.Event`, a thread-safe flag used to signal "please stop after the current item" across the boundary between the async event loop and the plain OS thread the actual sync work runs on.
- `import time` — used to measure how long each cycle takes, for logging.
- `from dataclasses import dataclass` — imports the decorator for the `SyncCycleResult` type below.
- `from qdrant_client import QdrantClient` — the client type used to talk to the Qdrant vector database; passed in rather than constructed here so this module doesn't own the database connection's lifecycle.
- `from agentic_rag.config import Settings` — the app's central configuration object (see project convention of using `pydantic-settings` rather than scattered `os.environ` reads), bundling things like the watched folder path, chunk size, sync interval, and Qdrant collection name.
- `from agentic_rag.embedding.cache import EmbeddingCache` — a cache used to avoid redundant embedding calls to Ollama within a single cycle.
- `from agentic_rag.indexing.upsert import delete_document, index_document` — the two functions that actually talk to Qdrant: one adds/updates a document's chunks in the index, the other removes a document's entries by path.
- `from agentic_rag.ingestion.pipeline import IngestionFailure` — the failure-record type shared across the ingestion pipeline, reused here for indexing and deletion failures too (so all three failure kinds have the same shape).
- `from agentic_rag.ingestion.snapshot_store import save_snapshot` — persists the folder snapshot to disk so it survives a process restart (see `ingestion/snapshot_store.py`).
- `from agentic_rag.ingestion.sync import sync_folder` — the function that detects folder changes and runs them through conversion/chunking/tagging/validation (see `ingestion/sync.py`).
- `from agentic_rag.ingestion.watcher import FileState` — the fingerprint type used to represent the snapshot.
- `from agentic_rag.observability.sync_log import log_sync_cycle, log_sync_cycle_error` — structured logging functions used to record what happened each cycle (or that a cycle failed outright).

### Lines 20-37 — `SyncCycleResult`
```python
@dataclass(frozen=True)
class SyncCycleResult:
    """Outcome of one `run_sync_cycle()` call: what actually reached the
    index, as opposed to `SyncResult` (`ingestion/sync.py`), which only
    reports what *changed on disk*.

    `indexing_failures` and `deletion_failures` are kept separate, not
    merged into one list - a document that failed to index and a
    deletion that failed are different failure kinds a caller may want to
    alert on differently, and merging them would make that distinction
    unrecoverable from the result alone.
    """

    indexed: list[str]
    deleted: list[str]
    ingestion_failures: list[IngestionFailure]
    indexing_failures: list[IngestionFailure]
    deletion_failures: list[IngestionFailure]
```
- `@dataclass(frozen=True)` — another immutable result record, following the pattern used throughout ingestion.
- The docstring draws a precise distinction from `SyncResult` (in `ingestion/sync.py`): that type reports what *changed on disk* (regardless of whether it made it into the index), while `SyncCycleResult` reports what actually *reached the index* — a document can be detected as changed but still fail to make it into Qdrant (e.g. an Ollama timeout while embedding it), and this type is what captures that further outcome.
- It also explains why indexing failures and deletion failures are kept as two separate lists instead of one combined list: a caller monitoring this system might want to alert differently on "we failed to add/update a document" versus "we failed to remove a deleted document," and merging them into one undifferentiated list would throw away that distinction with no way to recover it later.
- `indexed: list[str]` — relative paths of documents successfully written to the index this cycle.
- `deleted: list[str]` — relative paths successfully removed from the index this cycle.
- `ingestion_failures: list[IngestionFailure]` — files that failed during the ingestion pipeline stage itself (conversion, chunking, tagging, validation) — inherited straight from `sync_folder()`'s own failures.
- `indexing_failures: list[IngestionFailure]` — documents that ingested fine but failed when actually being written into Qdrant.
- `deletion_failures: list[IngestionFailure]` — deletions that were detected but failed when actually being removed from Qdrant.

### Lines 40-46 — `run_sync_cycle` signature
```python
def run_sync_cycle(
    *,
    settings: Settings,
    client: QdrantClient,
    previous_snapshot: dict[str, FileState],
    stop_event: threading.Event | None = None,
) -> tuple[SyncCycleResult, dict[str, FileState]]:
```
- `def run_sync_cycle(...)` — runs exactly one sync cycle: detect changes, index them, delete removed ones, and compute what the next cycle's baseline snapshot should be.
- `*,` — forces every parameter after this point to be passed by keyword only (e.g. `run_sync_cycle(settings=..., client=...)`), not positionally. This is a defensive readability choice: with five parameters of similar-looking types (several dicts and lists elsewhere in this codebase), forcing keyword arguments prevents a caller from accidentally swapping two arguments of compatible type in the wrong order.
- `settings: Settings` — the full application settings object, from which this function pulls the watched folder path, chunk size, access tiers, Qdrant collection name, embedding model, Ollama URL, sparse embedding model, and embedding timeout — all as attributes of one object rather than as a dozen separate parameters (explained further in the docstring below).
- `client: QdrantClient` — the already-connected Qdrant client to index into/delete from.
- `previous_snapshot: dict[str, FileState]` — the snapshot from the last time this ran, used as the diffing baseline.
- `stop_event: threading.Event | None = None` — an optional cooperative-cancellation flag; if set partway through the cycle, the function stops processing further items early (explained in the docstring below). Defaults to `None`, meaning "run to completion, no early stop requested" — useful for callers (like tests) that don't need cancellation support.
- `-> tuple[SyncCycleResult, dict[str, FileState]]:` — returns both the outcome of the cycle and the snapshot that should be used as `previous_snapshot` for the *next* call.

### Lines 47-100 — `run_sync_cycle` docstring
```python
    """One full sync cycle: detect changes since `previous_snapshot`, then
    propagate them to the Qdrant index (FR4). Composes `sync_folder()`
    (ingestion side: diff the watched folder, convert/chunk/tag) with
    `index_document()`/`delete_document()` (indexing side) - this is the
    one place an edit or deletion actually reaches the index, not just
    gets detected.

    Takes `settings` directly rather than the dozen individual keyword
    arguments `index_document()`/`sync_folder()` each already take - the
    same shape self-review originally flagged as worth avoiding for new
    code (`answer_with_cache()`'s old 19-parameter signature,
    PROJECT_TRACKER.md's Phase 7 log - since fixed by the same
    `settings`-object pattern this function already used): a typo'd kwarg
    name in a long hand-marshalled call is a silent bug waiting to happen,
    not caught until the affected setting is actually exercised.

    A fresh `EmbeddingCache` is created per cycle, not shared across
    cycles - a deliberate choice, not the default, since
    `docs/REQUIREMENTS.md` left this exact tradeoff open for Phase 7 to
    decide rather than assuming an answer. Chosen to bound memory at
    target scale (10,000+ docs): `sync_folder()`'s own diffing already
    skips re-embedding documents that didn't change between cycles
    regardless of the cache's lifetime, so the benefit a process-lifetime
    cache would add on top of that is narrower than it first looks, and
    not worth the unbounded growth risk over days/weeks of uptime.

    **A document or deletion that fails is retried on the next cycle, not
    silently dropped forever.** `sync_folder()`'s own `current_snapshot`
    is computed purely from disk state (`ingestion/sync.py`) - it has no
    idea whether `index_document()`/`delete_document()` actually
    succeeded for any given path. Naively returning it unmodified as the
    next cycle's `previous_snapshot` would mean a path that failed once
    (a transient Ollama timeout, a Qdrant error) is now "seen" and never
    diffed as changed again, since its on-disk fingerprint never moves.
    To prevent that: every path that failed - at the ingestion-pipeline
    stage (`ingestion_failures`), the indexing stage
    (`indexing_failures`), or the deletion stage (`deletion_failures`) -
    has its entry in the returned snapshot reverted to whatever it was in
    `previous_snapshot` (or removed entirely if it wasn't present there),
    so the next cycle's diff sees a mismatch and retries it. This applies
    uniformly to a failed edit, a failed new-file ingestion, and a failed
    deletion - re-inserting a deleted path's *old* fingerprint is exactly
    what makes the next diff report it as still-deleted, since the real
    file is (still) absent from disk either way.

    `stop_event`, if given, is checked between each document/deletion
    (not mid-item) - if set, every remaining, not-yet-attempted path is
    treated the same as a failure for the retry logic above. This exists
    so `run_sync_loop()` can request a graceful, bounded stop during
    shutdown: cancellation can't interrupt a thread that's already
    running (Python threads aren't preemptible), but checking between
    items bounds how much *additional* work happens after a stop is
    requested to roughly one item's worth, not the rest of the cycle.
    """
```
- The docstring is long because this function carries several non-obvious design decisions worth spelling out plainly:
  - **What it composes:** it's the glue between `sync_folder()` (which only detects and processes changes, without touching Qdrant) and `index_document()`/`delete_document()` (which only talk to Qdrant, without knowing about the filesystem) — this function is the one place both sides meet, i.e. the one place a real edit or deletion actually reaches the searchable index.
  - **Why it takes a `settings` object instead of many separate parameters:** the codebase had previously run into a real problem (referenced by name: `answer_with_cache()`'s old 19-parameter signature, tracked in `PROJECT_TRACKER.md`) where a long list of individually-passed keyword arguments made it easy to introduce a silent bug by mistyping one kwarg name in a hand-written call — a mistake that wouldn't be caught until that specific setting happened to matter at runtime. Passing one cohesive `settings` object avoids that whole category of bug for new code like this.
  - **Why a fresh `EmbeddingCache` per cycle, not a long-lived one:** this was a deliberate, considered choice rather than a default the author didn't think about — the project's `docs/REQUIREMENTS.md` explicitly left the cache-lifetime tradeoff open for this phase of work to decide. The reasoning given: since `sync_folder()` already avoids re-embedding files that haven't changed (via its own snapshot diffing), a cache that persists *across* cycles would only help with the (already-handled) case of unchanged documents, while a cache that grows forever across days or weeks of uptime is a real, unbounded memory risk at the target scale of 10,000+ documents. So the benefit of a longer-lived cache is smaller than it looks, and not worth the risk.
  - **The retry-on-failure design (the most important part):** `sync_folder()`'s returned `current_snapshot` reflects only what's physically on disk right now — it has no awareness of whether the *indexing* step (which happens later, in this function) actually succeeded for any given file. If this function just used that raw snapshot unmodified as the starting point for next cycle, then a file that failed to index (say, because Ollama briefly timed out) would incorrectly be "marked as seen" going forward — since its on-disk fingerprint hasn't changed, the next cycle's diff would never flag it as different, and it would never be retried, effectively vanishing from the index forever due to one transient hiccup. The fix: for every path that failed at any of the three possible stages (ingestion pipeline, indexing, or deletion), that path's entry in the snapshot handed back to the caller is reset back to whatever it was in `previous_snapshot` — or removed entirely if it wasn't there before. That mismatch between what's now recorded as "last known state" and what's actually on disk is exactly what makes the next cycle's diff see it as changed again and retry it. The same logic elegantly covers failed deletions too: putting the deleted file's *old* fingerprint back into the snapshot makes the next diff conclude "this still needs to be deleted," since the file remains genuinely absent from disk regardless of whether the deletion succeeded against Qdrant.
  - **How `stop_event` interacts with all of this:** it's checked only *between* items, not in the middle of processing one, because a currently-running item can't safely be interrupted partway (and Python threads can't be forcibly preempted anyway). Any path not yet attempted when the stop is noticed is treated exactly like a failure for the retry bookkeeping above, guaranteeing it gets picked up again next cycle. This bounds how much extra work happens after a shutdown is requested to roughly one item's worth — good enough for a fast, graceful stop without needing hard interruption.

### Line 101 — Creating the embedding cache
```python
    embedding_cache = EmbeddingCache()
```
- Creates a fresh, empty embedding cache scoped to just this one cycle, per the reasoning in the docstring above.

### Lines 103-108 — Running the ingestion side
```python
    sync_result = sync_folder(
        settings.watched_folder_path,
        previous_snapshot,
        settings.chunk_size_chars,
        settings.access_tiers,
    )
```
- Calls `sync_folder()` (from `ingestion/sync.py`) with the relevant settings pulled out of the `settings` object: the folder being watched, the previous snapshot to diff against, the target chunk size, and the configured list of valid access tiers. This performs the entire "detect + convert + chunk + tag + validate" ingestion step and returns a `SyncResult` bundling the new snapshot, the successfully processed documents, any ingestion-stage failures, and the list of deleted paths.

### Lines 110-112 — Seeding the "unresolved" tracking list
```python
    unresolved_paths: list[str] = [
        failure.relative_path for failure in sync_result.failures
    ]
```
- Starts a running list of every path that has failed (or, later, was skipped due to a stop request) somewhere in this cycle. It's seeded immediately with the paths that already failed during ingestion itself (before indexing was even attempted), since those also need their snapshot entries reverted at the end, per the retry design explained in the docstring.

### Lines 114-139 — Indexing successfully-ingested documents
```python
    indexed: list[str] = []
    indexing_failures: list[IngestionFailure] = []
    documents = sync_result.documents
    for position, document in enumerate(documents):
        if stop_event is not None and stop_event.is_set():
            unresolved_paths.extend(d.relative_path for d in documents[position:])
            break
        try:
            index_document(
                client,
                settings.qdrant_collection_name,
                document,
                embedding_model=settings.embedding_model,
                ollama_base_url=settings.ollama_base_url,
                sparse_model=settings.sparse_embedding_model,
                embedding_timeout_seconds=settings.embedding_timeout_seconds,
                embedding_cache=embedding_cache,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one bad document, see docstring
            reason = f"{type(exc).__name__}: {exc}"
            indexing_failures.append(
                IngestionFailure(relative_path=document.relative_path, reason=reason)
            )
            unresolved_paths.append(document.relative_path)
            continue
        indexed.append(document.relative_path)
```
- `indexed: list[str] = []` / `indexing_failures: list[IngestionFailure] = []` — accumulators for successful and failed indexing attempts this cycle.
- `documents = sync_result.documents` — the list of already-ingested documents (converted, chunked, tagged, validated) that now need to actually be written into Qdrant.
- `for position, document in enumerate(documents):` — iterates through them, tracking each one's position in the list (needed below to know which ones remain if a stop is requested partway through).
- `if stop_event is not None and stop_event.is_set():` — before attempting each document, checks whether a graceful stop has been requested.
- `unresolved_paths.extend(d.relative_path for d in documents[position:])` — if so, every document from the current position onward (including the current one, which hasn't been attempted yet) is added to `unresolved_paths`, so its snapshot entry gets reverted later and it's retried next cycle.
- `break` — stops the loop immediately, without attempting any more documents this cycle.
- `try: index_document(...)` — attempts to actually write this document's chunks into Qdrant, passing the client, the collection name, the document itself, and a handful of settings needed for embedding (which model to use, where Ollama is running, which sparse/keyword embedding model to use, how long to wait before timing out, and the shared cache for this cycle).
- `except Exception as exc: # noqa: BLE001 - isolate one bad document, see docstring` — catches any exception from indexing this one document. The `# noqa: BLE001` comment suppresses a linter warning (Ruff's "blind except" rule, which normally flags catching the broad `Exception` class) — with an inline justification rather than silently ignoring the warning, following the project's rule that no warning gets suppressed without an explanation, and pointing back to the function's own docstring for the full reasoning (isolating one bad document from crashing the whole cycle).
- `reason = f"{type(exc).__name__}: {exc}"` — builds a readable failure reason string.
- `indexing_failures.append(IngestionFailure(relative_path=document.relative_path, reason=reason))` — records this specific indexing failure.
- `unresolved_paths.append(document.relative_path)` — also adds it to the unresolved list so its snapshot entry gets reverted at the end (ensuring a retry next cycle).
- `continue` — moves on to the next document.
- `indexed.append(document.relative_path)` — if `index_document()` didn't raise, the document made it into the index successfully, and its path is recorded as indexed.

### Lines 141-157 — Deleting removed documents from the index
```python
    deleted: list[str] = []
    deletion_failures: list[IngestionFailure] = []
    deleted_paths = sync_result.deleted
    for position, relative_path in enumerate(deleted_paths):
        if stop_event is not None and stop_event.is_set():
            unresolved_paths.extend(deleted_paths[position:])
            break
        try:
            delete_document(client, settings.qdrant_collection_name, relative_path)
        except Exception as exc:  # noqa: BLE001 - isolate one bad deletion, see docstring
            reason = f"{type(exc).__name__}: {exc}"
            deletion_failures.append(
                IngestionFailure(relative_path=relative_path, reason=reason)
            )
            unresolved_paths.append(relative_path)
            continue
        deleted.append(relative_path)
```
- This block mirrors the indexing loop above exactly, but for the paths that `sync_folder()` detected as deleted from disk.
- `deleted: list[str] = []` / `deletion_failures: list[IngestionFailure] = []` — accumulators for successful and failed deletions.
- `deleted_paths = sync_result.deleted` — the relative paths that need to be removed from the index.
- `for position, relative_path in enumerate(deleted_paths):` — iterates through them, again tracking position for the same stop-handling reason as before.
- `if stop_event is not None and stop_event.is_set(): unresolved_paths.extend(deleted_paths[position:]); break` — same graceful-stop handling: any not-yet-attempted deletion is marked unresolved and the loop ends early.
- `try: delete_document(client, settings.qdrant_collection_name, relative_path)` — attempts to remove this path's entries from the Qdrant collection.
- `except Exception as exc: # noqa: BLE001 - isolate one bad deletion, see docstring` — same broad-catch pattern as indexing, with the same inline justification style, so one deletion that fails (e.g. a transient Qdrant error) doesn't stop the rest of the deletions from being attempted.
- `reason = ...` / `deletion_failures.append(...)` / `unresolved_paths.append(relative_path)` / `continue` — records the failure and ensures it gets retried next cycle, same pattern as before.
- `deleted.append(relative_path)` — if the deletion succeeded, records it as done.

### Lines 159-165 — Building the "carry forward" snapshot with reverted failures
```python
    carry_forward_snapshot = dict(sync_result.current_snapshot)
    for path in unresolved_paths:
        if path in previous_snapshot:
            carry_forward_snapshot[path] = previous_snapshot[path]
        else:
            carry_forward_snapshot.pop(path, None)
```
- `carry_forward_snapshot = dict(sync_result.current_snapshot)` — starts from a *copy* of the raw, disk-truth snapshot that `sync_folder()` computed (copying rather than mutating the original, so nothing else holding a reference to `sync_result.current_snapshot` is affected).
- `for path in unresolved_paths:` — walks through every path that failed or was skipped anywhere in this cycle (ingestion, indexing, or deletion).
- `if path in previous_snapshot: carry_forward_snapshot[path] = previous_snapshot[path]` — if this path existed in the *previous* cycle's snapshot, its entry in the snapshot being handed to the next cycle is reset back to that old fingerprint — undoing whatever the fresh disk scan just recorded for it. This is the mechanism, explained in the docstring, that makes the next cycle's diff see this path as "still different from last known good state" and therefore retry it.
- `else: carry_forward_snapshot.pop(path, None)` — if the path wasn't in the previous snapshot at all (meaning it's a brand-new file that failed on its very first ingestion attempt), its entry is removed from the carry-forward snapshot entirely (the `None` default to `.pop()` avoids an error if it's somehow already absent) — so the next cycle's diff sees it as still "created" and retries the whole ingestion from scratch.

### Lines 166-173 — Assembling and returning the result
```python
    result = SyncCycleResult(
        indexed=indexed,
        deleted=deleted,
        ingestion_failures=sync_result.failures,
        indexing_failures=indexing_failures,
        deletion_failures=deletion_failures,
    )
    return result, carry_forward_snapshot
```
- Packages everything computed during this cycle into a `SyncCycleResult`: what got indexed, what got deleted, and the three separate failure lists (ingestion-stage failures straight from `sync_folder()`, plus this function's own indexing and deletion failures).
- `return result, carry_forward_snapshot` — returns both the cycle's outcome and the corrected snapshot that the caller should use as the starting point for the *next* cycle (with failed/skipped paths reverted, as described above).

### Lines 176-181 — `run_sync_loop` signature
```python
async def run_sync_loop(
    *,
    settings: Settings,
    client: QdrantClient,
    initial_snapshot: dict[str, FileState] | None = None,
) -> None:
```
- `async def run_sync_loop(...)` — declared as an `async` coroutine (rather than a plain function) because it needs to run forever *alongside* the FastAPI application's own request handling in the same process, cooperatively yielding control rather than blocking.
- `*,` — again forces keyword-only arguments for the same clarity/safety reason as `run_sync_cycle`.
- `settings: Settings` — same settings object, used here for the sync interval, snapshot file path, and passed through to each `run_sync_cycle()` call.
- `client: QdrantClient` — the Qdrant client shared across every cycle this loop runs.
- `initial_snapshot: dict[str, FileState] | None = None` — the snapshot to start from; if not given, defaults to `None`, and is then treated as an empty dictionary (see below) — representing "we have no memory of previous state," e.g. a fresh corpus.
- `-> None:` — this coroutine is meant to run forever (until cancelled) and doesn't return a meaningful value.

### Lines 182-254 — `run_sync_loop` docstring
```python
    """Run `run_sync_cycle()` forever, every `settings.sync_interval_seconds`,
    until cancelled.

    Starts from `initial_snapshot` (defaults to `{}` - a corpus with no
    persisted snapshot yet treats every file already in the watched
    folder as new, indexing it once). The snapshot each cycle returns is
    persisted to `settings.sync_snapshot_path`
    (`ingestion/snapshot_store.py`) and becomes the next cycle's
    `previous_snapshot` - both so a restart can resume incrementally
    instead of re-walking and re-indexing the whole corpus, and, more
    importantly, so a file deleted while the process was down is still
    correctly detected as deleted on the next cycle after restart (an
    empty starting snapshot can never report anything as deleted, since
    `diff_snapshots()` only reports a path missing from `current` that
    was *present* in `previous`).

    Each cycle's blocking work (a filesystem walk, `markitdown`
    conversion, Ollama embedding calls, Qdrant upserts) runs via
    `asyncio.to_thread()` so it doesn't block the event loop `POST /query`
    request handling shares with this loop, in the same process - Qdrant's
    embedded/on-disk-locked mode (`docs/REQUIREMENTS.md` §9) means this
    can't run as a separate worker process without lock contention, so it
    has to share this one.

    **Cancellation waits for the in-flight cycle to actually stop, not
    just for the coroutine wrapper to unwind.** Cancelling an
    `asyncio.to_thread()` call only delivers `CancelledError` to the
    *awaiting coroutine*; whether the underlying OS thread's work is
    actually interrupted depends on timing, and naively trusting either
    outcome is wrong. Naively letting `CancelledError` propagate
    immediately would let `app.py`'s `lifespan` proceed to `client.close()`
    while an orphaned thread is still mid-`index_document()`/
    `delete_document()` against that same `client` - a real use-after-close
    race, not just a slow shutdown. The reverse mistake is just as real: an
    earlier version of this function waited on a plain `threading.Event`
    set in the cycle's own `finally` block - which deadlocked forever
    whenever cancellation landed *before* the work had actually started
    running (a genuinely common case for a fast, no-op cycle), because
    `concurrent.futures.Future.cancel()` succeeds for not-yet-started work
    and the cycle - and its `finally` - then never runs at all, so nothing
    would ever set that event. `asyncio.shield()` solves both failure modes
    at once: it prevents the wrapped future from ever actually being
    cancelled, whether it was already running or still queued, so it's
    always safe to wait for its real result. On cancellation, a
    `stop_event` shared with the in-flight `run_sync_cycle()` call is set
    (so it stops after its *current* item rather than continuing through
    the rest of the cycle, bounding shutdown to roughly one item's worth of
    work), then the same future is awaited again - shielded again, so this
    second wait can't itself be cancelled and fall into the same trap -
    before `CancelledError` is allowed to propagate.

    A whole cycle raising (rare - every per-document/deletion/failure is
    already isolated inside `run_sync_cycle()` itself) is caught and
    logged rather than killing the loop. Deliberately broad
    (`except Exception`, not a curated tuple of known-transient error
    types the way `plan_and_retrieve()`'s retry loop narrows its own
    catch) - unlike that loop, which is bounded (a handful of attempts,
    then a canonical fallback), this one runs for the entire life of the
    process with no bound, so resilience against an *unanticipated*
    failure mode matters more here than distinguishing "worth retrying"
    from "never will be": a masked bug still surfaces loudly in the logs
    via `log_sync_cycle_error()` (`observability/sync_log.py`, logged at
    `ERROR` with the real traceback attached), it just doesn't get to
    take down index freshness for the rest of the process's uptime along
    with it.

    Each cycle is timed via `time.monotonic()` and logged as one
    structured JSON line - `log_sync_cycle()` for a cycle that ran to
    completion (only when it actually changed something or failed, not
    on a resting no-op cycle, matching this loop's pre-existing
    behavior, to avoid one log line every `sync_interval_seconds`
    forever), or `log_sync_cycle_error()` when the whole cycle raised.
    """
```
- This is the densest docstring in the file; it explains four separate design decisions:
  - **Cold-start and persistence:** if no snapshot was ever saved, the loop starts as if the folder is entirely empty (`{}`), so the very first cycle treats every existing file as newly created and indexes it once — the same cold-start convention used by `snapshot_store.load_snapshot()`. After every cycle, the resulting snapshot is saved to disk (`settings.sync_snapshot_path`) and reused as the baseline for the *next* cycle. This persistence matters for two reasons: it lets a restarted process resume incrementally instead of redoing a full, expensive re-index of everything, and — the more important reason — it preserves the ability to detect deletions that happened while the process was offline, since (as explained in `snapshot_store.py`) a diff against an empty snapshot can never report anything as deleted.
  - **Why the actual work runs on a background thread:** each cycle does real, blocking I/O work (walking the filesystem, running `markitdown` conversions, calling Ollama for embeddings, writing to Qdrant) — none of that is natively `async`-friendly. Running it directly inside the coroutine would freeze the same event loop that's simultaneously trying to handle incoming `POST /query` HTTP requests, since they share one process. `asyncio.to_thread()` offloads that blocking work onto a separate OS thread so the event loop stays responsive. It also explains *why* this all has to share one process instead of being split into a separate worker process: Qdrant's embedded/on-disk-locked mode (referenced from the project's requirements doc) means only one process can hold the lock on the database files at a time, so running ingestion as a separate process would fight the web server for that lock.
  - **Careful, race-free cancellation on shutdown (the most involved part):** cancelling an `asyncio.to_thread()` call only cancels the *waiting coroutine*, not necessarily the underlying thread's actual work — so you can't just assume cancellation instantly stops everything. If the code let `CancelledError` propagate right away, the app's shutdown sequence (`app.py`'s `lifespan`) could proceed to closing the Qdrant client connection while a background thread is still in the middle of using that very client — a genuine use-after-close bug, not just an unusually slow shutdown. But the docstring also documents a previous, opposite mistake made and then fixed: an earlier version waited on a plain `threading.Event` set inside the cycle's own `finally` block, which could deadlock forever if cancellation arrived before the cycle's thread had even started running yet — because `concurrent.futures.Future.cancel()` can successfully cancel not-yet-started work outright, meaning the cycle body (and its `finally` block) would simply never run at all, so the event that shutdown was waiting on would never get set. The fix used here is `asyncio.shield()`, which prevents the wrapped future from ever truly being cancelled regardless of whether it's already running or still queued — so it's always safe to wait for its real, eventual result. The shutdown sequence used is: set the shared `stop_event` (telling the in-flight `run_sync_cycle()` to stop after its current item, bounding extra shutdown work to about one item), then await the same shielded future again (shielded a second time too, so this second wait can't itself be cancelled and fall into the same trap as before) — only after that does the function finally let `CancelledError` propagate up.
  - **Whole-cycle failure handling:** it's rare for an entire cycle to raise, since every individual document/deletion failure is already isolated inside `run_sync_cycle()` — but if something unanticipated does escape (a bug, not a known transient error), it's caught broadly (`except Exception`, not a narrow list of specific expected error types) and logged rather than crashing the loop. The docstring contrasts this with a different function elsewhere in the codebase (`plan_and_retrieve()`'s retry loop) that deliberately narrows its exception catching to known-transient error types — but explains that pattern doesn't fit here, because that other loop is bounded (a handful of attempts before giving up with a fallback), while this loop runs unboundedly for the entire life of the process, so guarding against *any* unexpected failure matters more than trying to distinguish "worth retrying" cases from others. Nothing is silently swallowed either way — the real exception and traceback are still logged loudly via `log_sync_cycle_error()` at `ERROR` level; it just doesn't get to permanently kill index freshness for the rest of the process's life.
  - **Structured logging per cycle:** each cycle's duration is measured, and a structured JSON log line is emitted — but only when something actually happened (a change, or a failure), not on every single idle cycle, to avoid flooding the logs with a "nothing happened" line every `sync_interval_seconds` forever. A cycle that finished is logged via `log_sync_cycle()`; a cycle that raised entirely is logged via `log_sync_cycle_error()`.

### Lines 272-273 — Initializing loop state before the loop starts
```python
    snapshot = initial_snapshot if initial_snapshot is not None else {}
    last_backup_time = time.monotonic()
```
- `snapshot = initial_snapshot if initial_snapshot is not None else {}` — sets the working `snapshot` variable to `initial_snapshot` if one was provided, or an empty dictionary otherwise — implementing the cold-start behavior described in the docstring. (Note: this local variable name `snapshot` shadows the imported `snapshot()` function from `watcher.py`, but that function is never called by name inside this particular function, so there's no actual conflict here.)
- `last_backup_time = time.monotonic()` — records "now" as the starting point the backup-interval check (further down the loop) measures elapsed time against. Setting this *before* the loop begins, rather than leaving it unset until the first backup, means the very first backup only happens once a full `qdrant_backup_interval_seconds` has passed after the process starts — there's no urgent need to back up an index moments after startup.

### Lines 274-285 — Starting one cycle as a shielded background task
```python
    while True:
        stop_event = threading.Event()
        cycle_start = time.monotonic()
        cycle_future = asyncio.ensure_future(
            asyncio.to_thread(
                run_sync_cycle,
                settings=settings,
                client=client,
                previous_snapshot=snapshot,
                stop_event=stop_event,
            )
        )
```
- `while True:` — the infinite loop that keeps this coroutine running cycle after cycle until it's cancelled from outside (e.g. during app shutdown).
- `stop_event = threading.Event()` — creates a brand-new stop flag for *this specific cycle*, shared between the coroutine (which may set it on cancellation) and the background thread running `run_sync_cycle` (which checks it between items).
- `cycle_start = time.monotonic()` — records the start time using `time.monotonic()` specifically (rather than wall-clock time like `time.time()`), because monotonic time is guaranteed to only move forward and is immune to system clock adjustments (like NTP corrections or daylight saving changes), making duration measurements reliable.
- `cycle_future = asyncio.ensure_future(asyncio.to_thread(run_sync_cycle, settings=settings, client=client, previous_snapshot=snapshot, stop_event=stop_event))` — schedules `run_sync_cycle()` to run on a separate background thread via `asyncio.to_thread()` (so it doesn't block the event loop), passing through all the needed arguments including this cycle's own `stop_event`. Wrapping the result with `asyncio.ensure_future()` turns it into a proper `Task` object that's already scheduled to start running immediately in the background, independent of whether/when it's awaited — this is important because it means the task begins executing right away, and can later be safely "shielded" and awaited (or re-awaited) without restarting or duplicating the work.

### Lines 269-274 — Waiting for the cycle, shielded from cancellation
```python
        try:
            result, snapshot = await asyncio.shield(cycle_future)
        except asyncio.CancelledError:
            stop_event.set()
            result, snapshot = await asyncio.shield(cycle_future)
            raise
```
- `try: result, snapshot = await asyncio.shield(cycle_future)` — waits for the cycle to finish, but wrapped in `asyncio.shield()`, which — per the docstring's explanation — ensures that if *this* awaiting coroutine gets cancelled, the underlying `cycle_future` task itself is protected from being cancelled too; it keeps running to completion regardless. On success, unpacks the cycle's result and its returned (possibly failure-reverted) snapshot, which becomes the new `snapshot` for the next iteration of the loop.
- `except asyncio.CancelledError:` — catches the case where this coroutine itself got cancelled (e.g. because the whole application is shutting down) while still waiting.
- `stop_event.set()` — signals the in-flight `run_sync_cycle()` (still running on its background thread) to stop after whatever item it's currently on, rather than continuing through the rest of the cycle — bounding how much extra work happens during shutdown.
- `result, snapshot = await asyncio.shield(cycle_future)` — waits *again*, still shielded, for the cycle to actually finish stopping and return its (now-shorter, stop_event-aware) result. This second shielded wait is what guarantees the shutdown sequence doesn't proceed (e.g. to closing the Qdrant client) until the background thread has genuinely stopped touching it — solving the use-after-close race described in the docstring.
- `raise` — after the cycle has genuinely finished, re-raises the original `CancelledError`, allowing cancellation to actually propagate now that it's safe to do so.

### Lines 275-279 — Handling an entire cycle raising
```python
        except Exception as exc:
            log_sync_cycle_error(
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.monotonic() - cycle_start,
            )
```
- `except Exception as exc:` — catches the rare case where the whole `run_sync_cycle()` call raised an exception instead of returning normally (deliberately broad, per the docstring's reasoning about this loop running unboundedly for the process's whole life).
- `log_sync_cycle_error(error=f"{type(exc).__name__}: {exc}", duration_seconds=time.monotonic() - cycle_start)` — logs the failure with a readable error description and how long the (failed) cycle took, using `log_sync_cycle_error()` from the observability module. Notably, `snapshot` is *not* updated in this branch — since the cycle never returned a new one, the loop simply retries with the same `previous_snapshot` it started this iteration with, on the next pass.

### Lines 280-305 — Handling a cycle that completed normally
```python
        else:
            save_snapshot(settings.sync_snapshot_path, snapshot)
            if (
                result.indexed
                or result.deleted
                or result.ingestion_failures
                or result.indexing_failures
                or result.deletion_failures
            ):
                log_sync_cycle(
                    indexed_count=len(result.indexed),
                    deleted_count=len(result.deleted),
                    ingestion_failure_count=len(result.ingestion_failures),
                    indexing_failure_count=len(result.indexing_failures),
                    deletion_failure_count=len(result.deletion_failures),
                    ingestion_failure_paths=[
                        f.relative_path for f in result.ingestion_failures
                    ],
                    indexing_failure_paths=[
                        f.relative_path for f in result.indexing_failures
                    ],
                    deletion_failure_paths=[
                        f.relative_path for f in result.deletion_failures
                    ],
                    duration_seconds=time.monotonic() - cycle_start,
                )
```
- `else:` — this `try`/`except`/`else` block's `else` clause runs only when the `try` completed with no exception at all (neither cancellation nor a raised error) — i.e., a genuinely successful cycle.
- `save_snapshot(settings.sync_snapshot_path, snapshot)` — persists the cycle's resulting snapshot to disk immediately (via `ingestion/snapshot_store.py`), so it survives a restart and becomes durable before the loop moves on.
- `if (result.indexed or result.deleted or result.ingestion_failures or result.indexing_failures or result.deletion_failures):` — checks whether *anything at all* happened this cycle — any successful indexing, deletion, or any failure of any of the three kinds. If every one of these is empty, the cycle was a no-op (nothing changed on disk since last time), and per the docstring's stated intent, no log line is emitted for that case, to avoid flooding the logs with a repetitive "nothing happened" message every single interval forever.
- `log_sync_cycle(...)` — if something did happen, logs a structured summary: counts of documents indexed, deleted, and failures of each of the three kinds, plus the *specific paths* that failed at each stage (so an operator reading the logs can see exactly which files need attention, not just how many), and how long the cycle took.

### Lines 325-344 — Backing up Qdrant's storage on its own schedule
```python
        if time.monotonic() - last_backup_time >= settings.qdrant_backup_interval_seconds:
            backup_start = time.monotonic()
            try:
                backup_path = await asyncio.to_thread(
                    backup_qdrant_storage,
                    settings.qdrant_storage_path,
                    settings.qdrant_backup_path,
                    retention_count=settings.qdrant_backup_retention_count,
                )
            except Exception as exc:  # noqa: BLE001 - a failed backup must not stall index freshness
                log_qdrant_backup_error(
                    error=f"{type(exc).__name__}: {exc}",
                    duration_seconds=time.monotonic() - backup_start,
                )
            else:
                log_qdrant_backup(
                    backup_path=str(backup_path),
                    duration_seconds=time.monotonic() - backup_start,
                )
            last_backup_time = time.monotonic()
```
This block runs every loop iteration (regardless of whether the sync cycle above found any changes, and regardless of whether it succeeded or raised), but the actual expensive work inside only happens rarely - see `indexing/backup.py` for the full reasoning behind why this exists (protecting against losing the whole search index, not a single document, since Qdrant's own backup feature doesn't work in the local/embedded mode this project runs).
- `if time.monotonic() - last_backup_time >= settings.qdrant_backup_interval_seconds:` — checks whether enough real time has passed since the last backup attempt (successful or not) to warrant trying again. `last_backup_time` starts out set to the loop's own start time (see the line right before `while True:` above), so the very first backup only happens once a full interval has elapsed after startup, not immediately. This check is deliberately independent of `sync_interval_seconds` (the sync cycle's own, usually much shorter, cadence) — copying the whole Qdrant storage folder on every ~60-second sync tick would be wasteful I/O.
- `backup_start = time.monotonic()` — records when this specific backup attempt began, for timing.
- `backup_path = await asyncio.to_thread(backup_qdrant_storage, settings.qdrant_storage_path, settings.qdrant_backup_path, retention_count=settings.qdrant_backup_retention_count)` — runs the actual backup (`indexing/backup.py::backup_qdrant_storage()`) in a separate thread via `asyncio.to_thread()`, the same technique the sync cycle itself uses above — copying a potentially large folder is a slow, blocking filesystem operation, and running it directly on the event loop would freeze `POST /query` request handling (which shares this same process/event loop) for however long the copy takes.
- `except Exception as exc: log_qdrant_backup_error(...)` — if the backup attempt raises for any reason (disk full, a permissions problem, anything), it's caught and logged as an error rather than allowed to propagate — a failed backup must never take down the sync loop's actual job (keeping the index fresh), the same reasoning already applied above to a whole sync cycle raising.
- `else: log_qdrant_backup(...)` — if the backup succeeded, logs its location and how long it took.
- `last_backup_time = time.monotonic()` — reset regardless of success or failure, so a persistently-failing backup retries at the same interval cadence instead of being attempted again on every single loop tick (which would spam the logs and hammer the disk with repeated failures).

### Line 346 — Sleeping until the next cycle
```python
        await asyncio.sleep(settings.sync_interval_seconds)
```
- Pauses this coroutine (without blocking the rest of the app, since `await` yields control back to the event loop) for `settings.sync_interval_seconds` before the `while True` loop begins its next iteration and starts another cycle. This is also a point where cancellation can be delivered cleanly, since the coroutine isn't holding any resources or mid-operation at this point.
