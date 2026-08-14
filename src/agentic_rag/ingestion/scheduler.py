from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.indexing.upsert import delete_document, index_document
from agentic_rag.ingestion.pipeline import IngestionFailure
from agentic_rag.ingestion.sync import sync_folder
from agentic_rag.ingestion.watcher import FileState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncCycleResult:
    """Outcome of one `run_sync_cycle()` call: what actually reached the
    index, as opposed to `SyncResult` (`ingestion/sync.py`), which only
    reports what *changed on disk*."""

    indexed: list[str]
    deleted: list[str]
    ingestion_failures: list[IngestionFailure]
    indexing_failures: list[IngestionFailure]


def run_sync_cycle(
    *,
    settings: Settings,
    client: QdrantClient,
    previous_snapshot: dict[str, FileState],
) -> tuple[SyncCycleResult, dict[str, FileState]]:
    """One full sync cycle: detect changes since `previous_snapshot`, then
    propagate them to the Qdrant index (FR4). Composes `sync_folder()`
    (ingestion side: diff the watched folder, convert/chunk/tag) with
    `index_document()`/`delete_document()` (indexing side) - this is the
    one place an edit or deletion actually reaches the index, not just
    gets detected.

    Takes `settings` directly rather than the dozen individual keyword
    arguments `index_document()`/`sync_folder()` each already take - the
    same shape self-review flagged as worth avoiding for new code
    (`answer_with_cache()`'s 19-parameter signature, PROJECT_TRACKER.md's
    Phase 7 log): a typo'd kwarg name in a long hand-marshalled call is a
    silent bug waiting to happen, not caught until the affected setting is
    actually exercised.

    A fresh `EmbeddingCache` is created per cycle, not shared across
    cycles - a deliberate choice, not the default, since
    `docs/REQUIREMENTS.md` left this exact tradeoff open for Phase 7 to
    decide rather than assuming an answer. Chosen to bound memory at
    target scale (10,000+ docs): `sync_folder()`'s own diffing already
    skips re-embedding documents that didn't change between cycles
    regardless of the cache's lifetime, so the benefit a process-lifetime
    cache would add on top of that - reusing identical chunk text (e.g.
    shared boilerplate) across documents that happen to change in
    *different* cycles - is narrower than it first looks, and not worth
    the unbounded growth risk over days/weeks of uptime.

    A document that fails at the indexing step (an embedding error, a
    Qdrant error) is caught and reported in `indexing_failures`, the same
    per-file isolation `sync_folder()`/`process_changes()` already apply
    to ingestion failures (`docs/REQUIREMENTS.md` §11): one bad document
    must not stall every other document - or every deletion - in the same
    cycle. A deletion that fails is isolated the same way.
    """
    embedding_cache = EmbeddingCache()

    sync_result = sync_folder(
        settings.watched_folder_path,
        previous_snapshot,
        settings.chunk_size_chars,
        settings.access_tiers,
    )

    indexed: list[str] = []
    indexing_failures: list[IngestionFailure] = []
    for document in sync_result.documents:
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
            indexing_failures.append(
                IngestionFailure(
                    relative_path=document.relative_path,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        indexed.append(document.relative_path)

    deleted: list[str] = []
    for relative_path in sync_result.deleted:
        try:
            delete_document(client, settings.qdrant_collection_name, relative_path)
        except Exception as exc:  # noqa: BLE001 - isolate one bad deletion, see docstring
            indexing_failures.append(
                IngestionFailure(
                    relative_path=relative_path,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        deleted.append(relative_path)

    result = SyncCycleResult(
        indexed=indexed,
        deleted=deleted,
        ingestion_failures=sync_result.failures,
        indexing_failures=indexing_failures,
    )
    return result, sync_result.current_snapshot


async def run_sync_loop(
    *,
    settings: Settings,
    client: QdrantClient,
    initial_snapshot: dict[str, FileState] | None = None,
) -> None:
    """Run `run_sync_cycle()` forever, every `settings.sync_interval_seconds`,
    until cancelled.

    Starts from `initial_snapshot` (defaults to `{}` - a fresh process
    treats every file already in the watched folder as new, indexing the
    whole corpus once on startup). The snapshot each cycle returns becomes
    the next cycle's `previous_snapshot`; nothing is persisted to disk
    between process restarts, so a restart re-walks and re-indexes the
    full corpus rather than resuming incrementally. That's wasteful, not
    wrong - `index_document()`'s point IDs are deterministic, so
    re-indexing an unchanged document just re-writes the same points, not
    duplicates - and is accepted as a known limitation for this phase
    rather than solved with on-disk snapshot persistence, matching
    `EmbeddingCache`'s own "not solved yet, flagged rather than silently
    deferred" precedent for a cost that only matters once restarts are
    frequent enough for it to.

    Each cycle's blocking work (a filesystem walk, `markitdown`
    conversion, Ollama embedding calls, Qdrant upserts) runs via
    `asyncio.to_thread()` so it doesn't block the event loop `POST /query`
    request handling shares with this loop, in the same process - Qdrant's
    embedded/on-disk-locked mode (`docs/REQUIREMENTS.md` §9) means this
    can't run as a separate worker process without lock contention, so it
    has to share this one.

    A whole cycle raising is caught and logged rather than killing the
    loop - every per-document/per-deletion failure is already isolated
    inside `run_sync_cycle()`, so this is a safety net for something
    outside that (e.g. `sync_folder()` itself failing to walk the folder).
    A transient failure should delay freshness, not permanently stop it:
    the next cycle naturally retries from the same `previous_snapshot`,
    since nothing was consumed or discarded on failure.
    """
    snapshot = initial_snapshot if initial_snapshot is not None else {}
    while True:
        try:
            result, snapshot = await asyncio.to_thread(
                run_sync_cycle, settings=settings, client=client, previous_snapshot=snapshot
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("sync cycle failed")
        else:
            if result.indexed or result.deleted or result.ingestion_failures or (
                result.indexing_failures
            ):
                logger.info(
                    "sync cycle: indexed=%d deleted=%d ingestion_failures=%d "
                    "indexing_failures=%d",
                    len(result.indexed),
                    len(result.deleted),
                    len(result.ingestion_failures),
                    len(result.indexing_failures),
                )
        await asyncio.sleep(settings.sync_interval_seconds)
