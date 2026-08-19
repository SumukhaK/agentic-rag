from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.indexing.backup import backup_qdrant_storage
from agentic_rag.indexing.upsert import delete_document, index_document
from agentic_rag.ingestion.pipeline import IngestionFailure
from agentic_rag.ingestion.snapshot_store import save_snapshot
from agentic_rag.ingestion.sync import sync_folder
from agentic_rag.ingestion.watcher import FileState
from agentic_rag.observability.backup_log import log_qdrant_backup, log_qdrant_backup_error
from agentic_rag.observability.sync_log import log_sync_cycle, log_sync_cycle_error


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


def run_sync_cycle(
    *,
    settings: Settings,
    client: QdrantClient,
    previous_snapshot: dict[str, FileState],
    stop_event: threading.Event | None = None,
) -> tuple[SyncCycleResult, dict[str, FileState]]:
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
    embedding_cache = EmbeddingCache()

    sync_result = sync_folder(
        settings.watched_folder_path,
        previous_snapshot,
        settings.chunk_size_chars,
        settings.access_tiers,
    )

    unresolved_paths: list[str] = [
        failure.relative_path for failure in sync_result.failures
    ]

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

    carry_forward_snapshot = dict(sync_result.current_snapshot)
    for path in unresolved_paths:
        if path in previous_snapshot:
            carry_forward_snapshot[path] = previous_snapshot[path]
        else:
            carry_forward_snapshot.pop(path, None)

    result = SyncCycleResult(
        indexed=indexed,
        deleted=deleted,
        ingestion_failures=sync_result.failures,
        indexing_failures=indexing_failures,
        deletion_failures=deletion_failures,
    )
    return result, carry_forward_snapshot


async def run_sync_loop(
    *,
    settings: Settings,
    client: QdrantClient,
    initial_snapshot: dict[str, FileState] | None = None,
) -> None:
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

    **A filesystem backup of Qdrant's on-disk storage runs on its own
    wall-clock cadence** (`settings.qdrant_backup_interval_seconds`,
    checked once per loop iteration, independent of `sync_interval_seconds`
    and of whether this iteration's cycle changed anything), not tied to
    cycle success/failure. This exists to bound *whole-index* loss - see
    `indexing/backup.py::backup_qdrant_storage()`'s own docstring for why
    Qdrant's native snapshot API isn't used (it doesn't support
    local/embedded mode) and why this matters at this project's target
    scale specifically (the real 10,000-document load test never finished
    a from-scratch rebuild on this hardware - see `loadtest/README.md`).
    A failed backup is logged (`log_qdrant_backup_error()`) and the loop
    continues rather than raising - the same "an unanticipated failure in
    one concern must not take down index freshness, the loop's actual
    job" reasoning already applied to a whole cycle raising, above.
    """
    snapshot = initial_snapshot if initial_snapshot is not None else {}
    last_backup_time = time.monotonic()
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

        try:
            result, snapshot = await asyncio.shield(cycle_future)
        except asyncio.CancelledError:
            stop_event.set()
            result, snapshot = await asyncio.shield(cycle_future)
            raise
        except Exception as exc:
            log_sync_cycle_error(
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.monotonic() - cycle_start,
            )
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

        if time.monotonic() - last_backup_time >= settings.qdrant_backup_interval_seconds:
            backup_start = time.monotonic()
            backup_future = asyncio.ensure_future(
                asyncio.to_thread(
                    backup_qdrant_storage,
                    settings.qdrant_storage_path,
                    settings.qdrant_backup_path,
                    retention_count=settings.qdrant_backup_retention_count,
                )
            )
            try:
                backup_path = await asyncio.shield(backup_future)
            except asyncio.CancelledError:
                # Same reasoning as the cycle above: cancelling this
                # coroutine can't stop the underlying OS thread mid-
                # shutil.copytree(), and letting CancelledError propagate
                # immediately would let this loop return - and the
                # caller proceed to client.close() - while that thread is
                # still reading/writing the same files the client is
                # about to close out from under it. Wait for the real
                # result (discarded; nothing to log during shutdown)
                # before re-raising.
                await asyncio.shield(backup_future)
                raise
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

        await asyncio.sleep(settings.sync_interval_seconds)
