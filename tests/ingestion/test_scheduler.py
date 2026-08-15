import asyncio
import itertools
import os
import threading
import time
from unittest.mock import patch

import pytest

from agentic_rag.config import Settings
from agentic_rag.embedding.sparse_client import SparseEmbeddingError, embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.indexing.upsert import _path_filter
from agentic_rag.ingestion.pipeline import IngestionFailure
from agentic_rag.ingestion.scheduler import SyncCycleResult, run_sync_cycle, run_sync_loop
from agentic_rag.ingestion.watcher import FileState, snapshot

KNOWN_TIERS = ["tier-1", "tier-2", "tier-3"]
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "documents"
TIER_1_A = os.path.join("tier-1", "a.txt")
TIER_1_B = os.path.join("tier-1", "b.txt")
TIER_1_GOOD = os.path.join("tier-1", "good.txt")


@pytest.fixture(scope="module", autouse=True)
def _require_sparse_model():
    try:
        embed_sparse_texts(["warmup"], model_name=SPARSE_MODEL)
    except SparseEmbeddingError as exc:
        pytest.skip(f"sparse embedding model unavailable: {exc}")


def _corpus(tmp_path):
    # Sibling of, not nested under, tmp_path/"qdrant" - Qdrant's own
    # on-disk storage files (.lock, meta.json, storage.sqlite) would
    # otherwise show up in the watched-folder snapshot as if they were
    # source documents.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return corpus


def _settings(corpus, qdrant_path) -> Settings:
    return Settings(
        watched_folder_path=corpus,
        qdrant_storage_path=qdrant_path,
        qdrant_collection_name=COLLECTION,
        sparse_embedding_model=SPARSE_MODEL,
        sync_snapshot_path=qdrant_path.parent / "sync_snapshot.json",
        _env_file=None,
    )


def _client(qdrant_path):
    client = get_client(qdrant_path)
    ensure_collection(client, collection_name=COLLECTION, vector_size=3)
    return client


def _count_points_for(client, relative_path):
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=_path_filter(relative_path),
        limit=100,
    )
    return len(points)


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_indexes_a_new_document(mock_embed_texts, tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    (corpus / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")

    result, _ = run_sync_cycle(
        settings=_settings(corpus, tmp_path / "qdrant"),
        client=client,
        previous_snapshot={},
    )

    assert result.indexed == [TIER_1_A]
    assert result.deleted == []
    assert result.ingestion_failures == []
    assert result.indexing_failures == []
    assert result.deletion_failures == []
    assert _count_points_for(client, TIER_1_A) == 1


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_deletes_a_removed_document(mock_embed_texts, tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    file_path = corpus / "tier-1" / "a.txt"
    file_path.write_text("Arsenal drew 1-1.")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")

    _, first_snapshot = run_sync_cycle(settings=settings, client=client, previous_snapshot={})
    file_path.unlink()

    result, _ = run_sync_cycle(
        settings=settings, client=client, previous_snapshot=first_snapshot
    )

    assert result.deleted == [TIER_1_A]
    assert result.indexed == []
    assert _count_points_for(client, TIER_1_A) == 0


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_returns_the_current_snapshot_for_the_next_cycle(
    mock_embed_texts, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    (corpus / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")

    _, current_snapshot = run_sync_cycle(
        settings=_settings(corpus, tmp_path / "qdrant"),
        client=client,
        previous_snapshot={},
    )

    assert current_snapshot == snapshot(corpus)


@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_isolates_an_ingestion_failure_from_indexing(
    mock_embed_texts, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    (corpus / "tier-1" / "good.txt").write_text("Arsenal drew 1-1.")
    (corpus / "bad.txt").write_text("no tier folder")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")

    result, _ = run_sync_cycle(
        settings=_settings(corpus, tmp_path / "qdrant"),
        client=client,
        previous_snapshot={},
    )

    assert result.indexed == [TIER_1_GOOD]
    assert [f.relative_path for f in result.ingestion_failures] == ["bad.txt"]


def test_run_sync_cycle_retries_a_persistently_failing_ingestion_on_every_cycle(tmp_path):
    # A file that fails ingestion (e.g. an untagged path) must not be
    # silently "seen" and dropped after the first cycle - sync_folder()'s
    # own current_snapshot is computed purely from disk state, with no
    # idea the file's ingestion failed, so naively carrying it forward
    # unmodified would make diff_snapshots() stop reporting it as changed
    # forever, even though nothing about it was ever actually resolved.
    corpus = _corpus(tmp_path)
    (corpus / "bad.txt").write_text("no tier folder")
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")

    first, first_snapshot = run_sync_cycle(
        settings=settings, client=client, previous_snapshot={}
    )
    second, _ = run_sync_cycle(
        settings=settings, client=client, previous_snapshot=first_snapshot
    )

    assert [f.relative_path for f in first.ingestion_failures] == ["bad.txt"]
    assert [f.relative_path for f in second.ingestion_failures] == ["bad.txt"]


@patch("agentic_rag.ingestion.scheduler.index_document")
def test_run_sync_cycle_isolates_an_indexing_failure_from_other_documents(
    mock_index_document, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    (corpus / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    (corpus / "tier-1" / "b.txt").write_text("Chelsea won 3-0.")
    mock_index_document.side_effect = [None, RuntimeError("embedding boom")]
    client = _client(tmp_path / "qdrant")

    result, _ = run_sync_cycle(
        settings=_settings(corpus, tmp_path / "qdrant"),
        client=client,
        previous_snapshot={},
    )

    assert result.indexed == [TIER_1_A]
    assert [f.relative_path for f in result.indexing_failures] == [TIER_1_B]
    assert "embedding boom" in result.indexing_failures[0].reason


@patch("agentic_rag.ingestion.scheduler.index_document")
def test_run_sync_cycle_retries_a_failed_document_on_the_next_cycle(
    mock_index_document, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    (corpus / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    mock_index_document.side_effect = RuntimeError("embedding boom")
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")

    first, first_snapshot = run_sync_cycle(
        settings=settings, client=client, previous_snapshot={}
    )
    # Nothing on disk changes between cycles - only the failure itself.
    second, _ = run_sync_cycle(
        settings=settings, client=client, previous_snapshot=first_snapshot
    )

    assert [f.relative_path for f in first.indexing_failures] == [TIER_1_A]
    assert [f.relative_path for f in second.indexing_failures] == [TIER_1_A]
    assert mock_index_document.call_count == 2


@patch("agentic_rag.ingestion.scheduler.delete_document")
@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_isolates_a_deletion_failure_from_other_deletions(
    mock_embed_texts, mock_delete_document, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    a_path = corpus / "tier-1" / "a.txt"
    b_path = corpus / "tier-1" / "b.txt"
    a_path.write_text("Arsenal drew 1-1.")
    b_path.write_text("Chelsea won 3-0.")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")
    _, first_snapshot = run_sync_cycle(settings=settings, client=client, previous_snapshot={})

    a_path.unlink()
    b_path.unlink()
    mock_delete_document.side_effect = [None, RuntimeError("qdrant boom")]

    result, _ = run_sync_cycle(
        settings=settings, client=client, previous_snapshot=first_snapshot
    )

    assert result.deleted == [TIER_1_A]
    assert [f.relative_path for f in result.deletion_failures] == [TIER_1_B]
    assert "qdrant boom" in result.deletion_failures[0].reason


@patch("agentic_rag.ingestion.scheduler.delete_document")
@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_retries_a_failed_deletion_on_the_next_cycle(
    mock_embed_texts, mock_delete_document, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    file_path = corpus / "tier-1" / "a.txt"
    file_path.write_text("Arsenal drew 1-1.")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")
    _, first_snapshot = run_sync_cycle(settings=settings, client=client, previous_snapshot={})

    file_path.unlink()
    mock_delete_document.side_effect = RuntimeError("qdrant boom")

    second, second_snapshot = run_sync_cycle(
        settings=settings, client=client, previous_snapshot=first_snapshot
    )
    # The file is still gone from disk - a naive implementation would
    # only ever report a path as "deleted" once (when it first vanishes
    # from the snapshot); retrying requires re-inserting its old
    # fingerprint into the carried-forward snapshot so the *next* diff
    # sees it as still-missing-from-current, not as "already handled."
    third, _ = run_sync_cycle(
        settings=settings, client=client, previous_snapshot=second_snapshot
    )

    assert [f.relative_path for f in second.deletion_failures] == [TIER_1_A]
    assert [f.relative_path for f in third.deletion_failures] == [TIER_1_A]
    assert mock_delete_document.call_count == 2


@patch("agentic_rag.ingestion.scheduler.index_document")
def test_run_sync_cycle_stops_indexing_early_when_stop_event_is_set(
    mock_index_document, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    (corpus / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    (corpus / "tier-1" / "b.txt").write_text("Chelsea won 3-0.")
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")
    stop_event = threading.Event()
    stop_event.set()

    result, next_snapshot = run_sync_cycle(
        settings=settings,
        client=client,
        previous_snapshot={},
        stop_event=stop_event,
    )

    assert result.indexed == []
    mock_index_document.assert_not_called()
    # Neither document was attempted - both must be retried, so neither
    # should appear in the carried-forward snapshot (they weren't in
    # previous_snapshot either, since this is their first appearance).
    assert TIER_1_A not in next_snapshot
    assert TIER_1_B not in next_snapshot


@patch("agentic_rag.ingestion.scheduler.delete_document")
@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_stops_deleting_early_when_stop_event_is_set(
    mock_embed_texts, mock_delete_document, tmp_path
):
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    a_path = corpus / "tier-1" / "a.txt"
    b_path = corpus / "tier-1" / "b.txt"
    a_path.write_text("Arsenal drew 1-1.")
    b_path.write_text("Chelsea won 3-0.")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")
    _, first_snapshot = run_sync_cycle(settings=settings, client=client, previous_snapshot={})

    a_path.unlink()
    b_path.unlink()
    stop_event = threading.Event()
    stop_event.set()

    result, next_snapshot = run_sync_cycle(
        settings=settings,
        client=client,
        previous_snapshot=first_snapshot,
        stop_event=stop_event,
    )

    assert result.deleted == []
    mock_delete_document.assert_not_called()
    # Both deletions must be retried next cycle - their old fingerprints
    # need to be back in the carried-forward snapshot so the next diff
    # still sees them as missing-from-current (deleted).
    assert next_snapshot.get(TIER_1_A) == first_snapshot[TIER_1_A]
    assert next_snapshot.get(TIER_1_B) == first_snapshot[TIER_1_B]


@patch("agentic_rag.ingestion.scheduler.EmbeddingCache")
@patch("agentic_rag.indexing.upsert.embed_texts")
def test_run_sync_cycle_uses_a_fresh_embedding_cache_each_call(
    mock_embed_texts, mock_embedding_cache_cls, tmp_path
):
    # docs/REQUIREMENTS.md explicitly left "one cache per cycle vs. one for
    # the process lifetime" open for Phase 7 to decide - a fresh cache per
    # cycle was chosen to bound memory at target scale (10,000+ docs)
    # rather than let a process-lifetime cache grow unbounded.
    corpus = _corpus(tmp_path)
    (corpus / "tier-1").mkdir()
    (corpus / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
    client = _client(tmp_path / "qdrant")
    settings = _settings(corpus, tmp_path / "qdrant")

    run_sync_cycle(settings=settings, client=client, previous_snapshot={})
    run_sync_cycle(settings=settings, client=client, previous_snapshot={})

    assert mock_embedding_cache_cls.call_count == 2


# --- run_sync_loop ---------------------------------------------------------
#
# No pytest-asyncio dependency: each test wraps its coroutine body in
# asyncio.run() from a plain sync test function, matching this codebase's
# established preference for stdlib-only solutions over a new dependency
# for what's otherwise a one-line need.


def _empty_result() -> SyncCycleResult:
    return SyncCycleResult(
        indexed=[],
        deleted=[],
        ingestion_failures=[],
        indexing_failures=[],
        deletion_failures=[],
    )


def _loop_settings(tmp_path) -> Settings:
    corpus = _corpus(tmp_path)
    return _settings(corpus, tmp_path / "qdrant")


@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_calls_run_sync_cycle_repeatedly(
    mock_run_cycle, mock_sleep, mock_save_snapshot, tmp_path
):
    mock_run_cycle.side_effect = [
        (_empty_result(), {}),
        (_empty_result(), {}),
    ]
    mock_sleep.side_effect = [None, asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    assert mock_run_cycle.call_count == 2


@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_continues_after_a_cycle_raises(
    mock_run_cycle, mock_sleep, mock_save_snapshot, tmp_path
):
    # A whole cycle raising should be exceedingly rare (every per-document
    # step is already isolated inside run_sync_cycle), but if it happens -
    # e.g. sync_folder() itself failing to walk a folder with a permissions
    # error - one bad cycle must not permanently stop freshness, the same
    # "one bad thing can't stall everything else" reasoning
    # docs/REQUIREMENTS.md §11 states for a single bad file.
    mock_run_cycle.side_effect = [RuntimeError("boom"), (_empty_result(), {})]
    mock_sleep.side_effect = [None, asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    assert mock_run_cycle.call_count == 2


@patch("agentic_rag.ingestion.scheduler.time.monotonic", side_effect=itertools.count(0, 1))
@patch("agentic_rag.ingestion.scheduler.log_sync_cycle")
@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_logs_a_completed_cycle_with_changes(
    mock_run_cycle, mock_sleep, mock_save_snapshot, mock_log_cycle, mock_monotonic, tmp_path
):
    # time.monotonic() is mocked to a deterministic counting sequence,
    # not left to the real clock - CLAUDE.md forbids real wall-clock
    # reliance in tests. duration_seconds still isn't asserted to an
    # exact value: asyncio's own internals also call time.monotonic() an
    # implementation-detail number of times around awaiting the shielded
    # cycle future, so only this function's own two calls (cycle
    # start/end) are guaranteed to be in the sequence, not which two -
    # an infinite counter (rather than a fixed-length list) tolerates
    # that extra, uncounted consumption without running out of values.
    result = SyncCycleResult(
        indexed=["a.md"],
        deleted=[],
        ingestion_failures=[],
        indexing_failures=[],
        deletion_failures=[],
    )
    mock_run_cycle.side_effect = [(result, {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    mock_log_cycle.assert_called_once()
    kwargs = mock_log_cycle.call_args.kwargs
    assert kwargs["indexed_count"] == 1
    assert kwargs["deleted_count"] == 0
    assert kwargs["ingestion_failure_count"] == 0
    assert kwargs["indexing_failure_count"] == 0
    assert kwargs["deletion_failure_count"] == 0
    assert kwargs["ingestion_failure_paths"] == []
    assert kwargs["indexing_failure_paths"] == []
    assert kwargs["deletion_failure_paths"] == []
    assert kwargs["duration_seconds"] >= 0.0


@patch("agentic_rag.ingestion.scheduler.log_sync_cycle")
@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_does_not_log_a_no_op_cycle(
    mock_run_cycle, mock_sleep, mock_save_snapshot, mock_log_cycle, tmp_path
):
    # A resting cycle that changed nothing must not produce a log line
    # every settings.sync_interval_seconds forever - matches this loop's
    # pre-existing behavior (before structured logging was added).
    mock_run_cycle.side_effect = [(_empty_result(), {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    mock_log_cycle.assert_not_called()


@patch("agentic_rag.ingestion.scheduler.time.monotonic", side_effect=itertools.count(0, 1))
@patch("agentic_rag.ingestion.scheduler.log_sync_cycle")
@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_logs_failure_paths_not_just_counts(
    mock_run_cycle, mock_sleep, mock_save_snapshot, mock_log_cycle, mock_monotonic, tmp_path
):
    result = SyncCycleResult(
        indexed=[],
        deleted=[],
        ingestion_failures=[],
        indexing_failures=[IngestionFailure(relative_path="tier-1/bad.md", reason="boom")],
        deletion_failures=[],
    )
    mock_run_cycle.side_effect = [(result, {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    kwargs = mock_log_cycle.call_args.kwargs
    assert kwargs["indexing_failure_paths"] == ["tier-1/bad.md"]


@patch("agentic_rag.ingestion.scheduler.time.monotonic", side_effect=itertools.count(0, 1))
@patch("agentic_rag.ingestion.scheduler.log_sync_cycle_error")
@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_logs_a_cycle_error(
    mock_run_cycle, mock_sleep, mock_save_snapshot, mock_log_error, mock_monotonic, tmp_path
):
    mock_run_cycle.side_effect = [RuntimeError("boom"), (_empty_result(), {})]
    mock_sleep.side_effect = [None, asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    mock_log_error.assert_called_once()
    kwargs = mock_log_error.call_args.kwargs
    assert kwargs["error"] == "RuntimeError: boom"
    assert kwargs["duration_seconds"] >= 0.0


@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_threads_the_snapshot_from_one_cycle_to_the_next(
    mock_run_cycle, mock_sleep, mock_save_snapshot, tmp_path
):
    second_snapshot = {"a.txt": FileState(size=1, mtime_ns=1)}
    mock_run_cycle.side_effect = [
        (_empty_result(), second_snapshot),
        (_empty_result(), second_snapshot),
    ]
    mock_sleep.side_effect = [None, asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object(), initial_snapshot={})

    asyncio.run(_run())

    first_kwargs = mock_run_cycle.call_args_list[0].kwargs
    second_kwargs = mock_run_cycle.call_args_list[1].kwargs
    assert first_kwargs["previous_snapshot"] == {}
    assert second_kwargs["previous_snapshot"] == second_snapshot


@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_starts_from_the_given_initial_snapshot(
    mock_run_cycle, mock_sleep, mock_save_snapshot, tmp_path
):
    initial_snapshot = {"a.txt": FileState(size=1, mtime_ns=1)}
    mock_run_cycle.side_effect = [(_empty_result(), {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(
                settings=settings, client=object(), initial_snapshot=initial_snapshot
            )

    asyncio.run(_run())

    assert mock_run_cycle.call_args_list[0].kwargs["previous_snapshot"] == initial_snapshot


@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_defaults_to_an_empty_initial_snapshot(
    mock_run_cycle, mock_sleep, mock_save_snapshot, tmp_path
):
    mock_run_cycle.side_effect = [(_empty_result(), {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    assert mock_run_cycle.call_args_list[0].kwargs["previous_snapshot"] == {}


@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_sleeps_for_settings_sync_interval_seconds(
    mock_run_cycle, mock_sleep, mock_save_snapshot, tmp_path
):
    mock_run_cycle.side_effect = [(_empty_result(), {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path).model_copy(update={"sync_interval_seconds": 42.0})

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    mock_sleep.assert_called_once_with(42.0)


@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_persists_the_snapshot_after_each_successful_cycle(
    mock_run_cycle, mock_sleep, tmp_path
):
    new_snapshot = {"a.txt": FileState(size=1, mtime_ns=1)}
    mock_run_cycle.side_effect = [(_empty_result(), new_snapshot)]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    with patch("agentic_rag.ingestion.scheduler.save_snapshot") as mock_save_snapshot:
        asyncio.run(_run())

    mock_save_snapshot.assert_called_once_with(settings.sync_snapshot_path, new_snapshot)


@patch("agentic_rag.ingestion.scheduler.save_snapshot")
@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_does_not_persist_a_snapshot_when_a_cycle_raises(
    mock_run_cycle, mock_sleep, mock_save_snapshot, tmp_path
):
    mock_run_cycle.side_effect = RuntimeError("boom")
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    mock_save_snapshot.assert_not_called()


def test_run_sync_loop_waits_for_the_in_flight_cycle_before_propagating_cancellation(
    tmp_path,
):
    # Cancelling a task awaiting asyncio.to_thread() only stops the
    # *awaiting coroutine* - Python threads can't be interrupted, so the
    # real work keeps running regardless. If run_sync_loop let
    # CancelledError propagate immediately, app.py's lifespan could
    # proceed to client.close() while this orphaned thread is still
    # mid-index_document()/delete_document() against that same client - a
    # real use-after-close race, reproduced independently by 3 review
    # angles for PR #46 and empirically confirmed with a standalone
    # asyncio.to_thread() cancellation script before this fix.
    cycle_finished = threading.Event()

    def fake_cycle(*, settings, client, previous_snapshot, stop_event):
        # Simulate a cycle that's genuinely still running (mid-item) when
        # cancellation arrives - it only notices stop_event and wraps up
        # after "finishing" its current item, the same way a real
        # in-flight index_document()/delete_document() call can't be
        # interrupted mid-call.
        stop_event.wait(timeout=5)
        time.sleep(0.2)
        cycle_finished.set()
        return _empty_result(), {}

    settings = _loop_settings(tmp_path)

    async def _run():
        with patch(
            "agentic_rag.ingestion.scheduler.run_sync_cycle", side_effect=fake_cycle
        ), patch("agentic_rag.ingestion.scheduler.save_snapshot"):
            task = asyncio.ensure_future(run_sync_loop(settings=settings, client=object()))
            await asyncio.sleep(0.1)  # let the loop actually start its first cycle
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # By the time `await task` raises, the in-flight cycle must
            # have genuinely finished - not just been asked to.
            assert cycle_finished.is_set()

    asyncio.run(_run())
