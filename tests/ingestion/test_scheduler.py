import asyncio
import os
from unittest.mock import patch

import pytest

from agentic_rag.config import Settings
from agentic_rag.embedding.sparse_client import SparseEmbeddingError, embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.indexing.upsert import _path_filter
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
    assert [f.relative_path for f in result.indexing_failures] == [TIER_1_B]
    assert "qdrant boom" in result.indexing_failures[0].reason


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
    return SyncCycleResult(indexed=[], deleted=[], ingestion_failures=[], indexing_failures=[])


def _loop_settings(tmp_path) -> Settings:
    corpus = _corpus(tmp_path)
    return _settings(corpus, tmp_path / "qdrant")


@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_calls_run_sync_cycle_repeatedly(mock_run_cycle, mock_sleep, tmp_path):
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


@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_continues_after_a_cycle_raises(mock_run_cycle, mock_sleep, tmp_path):
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


@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_threads_the_snapshot_from_one_cycle_to_the_next(
    mock_run_cycle, mock_sleep, tmp_path
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


@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_starts_from_the_given_initial_snapshot(
    mock_run_cycle, mock_sleep, tmp_path
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


@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_defaults_to_an_empty_initial_snapshot(
    mock_run_cycle, mock_sleep, tmp_path
):
    mock_run_cycle.side_effect = [(_empty_result(), {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path)

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    assert mock_run_cycle.call_args_list[0].kwargs["previous_snapshot"] == {}


@patch("agentic_rag.ingestion.scheduler.asyncio.sleep")
@patch("agentic_rag.ingestion.scheduler.run_sync_cycle")
def test_run_sync_loop_sleeps_for_settings_sync_interval_seconds(
    mock_run_cycle, mock_sleep, tmp_path
):
    mock_run_cycle.side_effect = [(_empty_result(), {})]
    mock_sleep.side_effect = [asyncio.CancelledError()]
    settings = _loop_settings(tmp_path).model_copy(update={"sync_interval_seconds": 42.0})

    async def _run():
        with pytest.raises(asyncio.CancelledError):
            await run_sync_loop(settings=settings, client=object())

    asyncio.run(_run())

    mock_sleep.assert_called_once_with(42.0)
