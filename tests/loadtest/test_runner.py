import json
from pathlib import Path
from unittest.mock import patch

from agentic_rag.config import Settings
from agentic_rag.ingestion.pipeline import IngestionFailure
from agentic_rag.ingestion.scheduler import SyncCycleResult
from agentic_rag.ingestion.watcher import FileState
from agentic_rag.loadtest.runner import (
    LoadTestReport,
    _copy_batch,
    _next_batch,
    _report_path_for,
    main,
    run_load_test,
)

# --- _next_batch --------------------------------------------------------


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_next_batch_returns_up_to_batch_size_unstaged_files(tmp_path):
    staged = tmp_path / "staged"
    watched = tmp_path / "watched"
    for i in range(5):
        _write(staged / "tier-1" / f"doc_{i:05d}.md")

    batch = _next_batch(staged, watched, batch_size=3)

    assert len(batch) == 3


def test_next_batch_excludes_files_already_present_in_watched_dir(tmp_path):
    staged = tmp_path / "staged"
    watched = tmp_path / "watched"
    _write(staged / "tier-1" / "doc_00000.md")
    _write(staged / "tier-1" / "doc_00001.md")
    _write(watched / "tier-1" / "doc_00000.md")

    batch = _next_batch(staged, watched, batch_size=10)

    assert [p.name for p in batch] == ["doc_00001.md"]


def test_next_batch_returns_empty_when_everything_already_copied(tmp_path):
    staged = tmp_path / "staged"
    watched = tmp_path / "watched"
    _write(staged / "tier-1" / "doc_00000.md")
    _write(watched / "tier-1" / "doc_00000.md")

    batch = _next_batch(staged, watched, batch_size=10)

    assert batch == []


def test_next_batch_returns_empty_when_watched_dir_does_not_exist_yet(tmp_path):
    staged = tmp_path / "staged"
    watched = tmp_path / "does_not_exist"
    _write(staged / "tier-1" / "doc_00000.md")

    batch = _next_batch(staged, watched, batch_size=10)

    assert len(batch) == 1


def test_next_batch_preserves_tier_subfolder_structure(tmp_path):
    staged = tmp_path / "staged"
    watched = tmp_path / "watched"
    _write(staged / "tier-2" / "doc_00000.md")

    batch = _next_batch(staged, watched, batch_size=10)

    assert batch[0].relative_to(staged) == Path("tier-2") / "doc_00000.md"


# --- _copy_batch ---------------------------------------------------------


def test_copy_batch_copies_files_preserving_tier_subfolder_structure(tmp_path):
    staged = tmp_path / "staged"
    watched = tmp_path / "watched"
    _write(staged / "tier-1" / "doc_00000.md", content="hello")
    batch = [staged / "tier-1" / "doc_00000.md"]

    _copy_batch(batch, staged, watched)

    assert (watched / "tier-1" / "doc_00000.md").read_text() == "hello"


# --- run_load_test --------------------------------------------------------


def _settings(tmp_path) -> Settings:
    return Settings(
        watched_folder_path=tmp_path / "corpus",
        qdrant_storage_path=tmp_path / "qdrant",
        sync_snapshot_path=tmp_path / "sync_snapshot.json",
        loadtest_corpus_staging_path=tmp_path / "loadtest_staging",
        loadtest_watched_folder_path=tmp_path / "loadtest_watched",
        loadtest_qdrant_storage_path=tmp_path / "loadtest_qdrant",
        loadtest_qdrant_collection_name="loadtest_documents",
        loadtest_sync_snapshot_path=tmp_path / "loadtest_snapshot.json",
        loadtest_results_path=tmp_path / "loadtest_results",
        loadtest_batch_size=2,
        _env_file=None,
    )


def _state(n: int = 1) -> FileState:
    return FileState(size=n, mtime_ns=n)


def _sync_result(indexed_count=2) -> SyncCycleResult:
    return SyncCycleResult(
        indexed=[f"tier-1/doc_{i:05d}.md" for i in range(indexed_count)],
        deleted=[],
        ingestion_failures=[],
        indexing_failures=[],
        deletion_failures=[],
    )


@patch("agentic_rag.loadtest.runner._run_query_latency_phase")
@patch("agentic_rag.loadtest.runner.run_sync_cycle")
def test_run_load_test_processes_every_staged_document_in_batches(
    mock_sync_cycle, mock_query_phase, tmp_path
):
    settings = _settings(tmp_path)
    for i in range(5):
        _write(settings.loadtest_corpus_staging_path / "tier-1" / f"doc_{i:05d}.md")
    mock_sync_cycle.side_effect = [
        (_sync_result(2), {"a": _state(1)}),
        (_sync_result(2), {"a": _state(1), "b": _state(2)}),
        (_sync_result(1), {"a": _state(1), "b": _state(2), "c": _state(3)}),
    ]
    mock_query_phase.return_value = [1.0, 2.0]

    report = run_load_test(settings=settings)

    assert mock_sync_cycle.call_count == 3
    assert report.total_indexed == 5
    assert report.batch_count == 3


@patch("agentic_rag.loadtest.runner._run_query_latency_phase")
@patch("agentic_rag.loadtest.runner.run_sync_cycle")
def test_run_load_test_saves_a_snapshot_checkpoint_after_every_batch(
    mock_sync_cycle, mock_query_phase, tmp_path
):
    settings = _settings(tmp_path)
    for i in range(4):
        _write(settings.loadtest_corpus_staging_path / "tier-1" / f"doc_{i:05d}.md")
    mock_sync_cycle.side_effect = [
        (_sync_result(2), {"a": _state(1)}),
        (_sync_result(2), {"a": _state(1), "b": _state(2)}),
    ]
    mock_query_phase.return_value = []

    run_load_test(settings=settings)

    assert json.loads(settings.loadtest_sync_snapshot_path.read_text()) == {
        "a": {"size": 1, "mtime_ns": 1},
        "b": {"size": 2, "mtime_ns": 2},
    }


@patch("agentic_rag.loadtest.runner._run_query_latency_phase")
@patch("agentic_rag.loadtest.runner.run_sync_cycle")
def test_run_load_test_only_indexes_against_the_dedicated_loadtest_collection(
    mock_sync_cycle, mock_query_phase, tmp_path
):
    settings = _settings(tmp_path)
    _write(settings.loadtest_corpus_staging_path / "tier-1" / "doc_00000.md")
    mock_sync_cycle.return_value = (_sync_result(1), {})
    mock_query_phase.return_value = []

    run_load_test(settings=settings)

    cycle_settings = mock_sync_cycle.call_args.kwargs["settings"]
    assert cycle_settings.watched_folder_path == settings.loadtest_watched_folder_path
    assert cycle_settings.qdrant_collection_name == settings.loadtest_qdrant_collection_name


@patch("agentic_rag.loadtest.runner._run_query_latency_phase")
@patch("agentic_rag.loadtest.runner.run_sync_cycle")
def test_run_load_test_counts_ingestion_and_indexing_failures_separately(
    mock_sync_cycle, mock_query_phase, tmp_path
):
    settings = _settings(tmp_path)
    _write(settings.loadtest_corpus_staging_path / "tier-1" / "doc_00000.md")
    mock_sync_cycle.return_value = (
        SyncCycleResult(
            indexed=[],
            deleted=[],
            ingestion_failures=[IngestionFailure(relative_path="tier-1/bad.md", reason="boom")],
            indexing_failures=[IngestionFailure(relative_path="tier-1/other.md", reason="oops")],
            deletion_failures=[],
        ),
        {},
    )
    mock_query_phase.return_value = []

    report = run_load_test(settings=settings)

    assert report.total_ingestion_failures == 1
    assert report.total_indexing_failures == 1


@patch("agentic_rag.loadtest.runner._run_query_latency_phase")
@patch("agentic_rag.loadtest.runner.run_sync_cycle")
def test_run_load_test_records_query_latencies_from_the_post_load_phase(
    mock_sync_cycle, mock_query_phase, tmp_path
):
    settings = _settings(tmp_path)
    _write(settings.loadtest_corpus_staging_path / "tier-1" / "doc_00000.md")
    mock_sync_cycle.return_value = (_sync_result(1), {})
    mock_query_phase.return_value = [0.5, 0.7, 0.6]

    report = run_load_test(settings=settings)

    assert report.query_latencies_seconds == [0.5, 0.7, 0.6]


@patch("agentic_rag.loadtest.runner._run_query_latency_phase")
@patch("agentic_rag.loadtest.runner.run_sync_cycle")
def test_run_load_test_resumes_without_recopying_already_watched_files(
    mock_sync_cycle, mock_query_phase, tmp_path
):
    # Simulates a restart after a crash: one file is already present in
    # the loadtest watched folder from a prior (interrupted) run.
    settings = _settings(tmp_path)
    _write(settings.loadtest_corpus_staging_path / "tier-1" / "doc_00000.md")
    _write(settings.loadtest_corpus_staging_path / "tier-1" / "doc_00001.md")
    _write(settings.loadtest_watched_folder_path / "tier-1" / "doc_00000.md")
    mock_sync_cycle.return_value = (_sync_result(1), {})
    mock_query_phase.return_value = []

    run_load_test(settings=settings)

    assert mock_sync_cycle.call_count == 1


@patch("agentic_rag.loadtest.runner.log_loadtest_batch")
@patch("agentic_rag.loadtest.runner._run_query_latency_phase")
@patch("agentic_rag.loadtest.runner.run_sync_cycle")
def test_run_load_test_logs_a_running_cumulative_total_not_just_the_batch_count(
    mock_sync_cycle, mock_query_phase, mock_log_batch, tmp_path
):
    # cumulative_indexed must be the running total across every batch so
    # far, not just the current batch's own count - otherwise a reader
    # tailing the log can never see real progress toward 10,000.
    settings = _settings(tmp_path)
    for i in range(4):
        _write(settings.loadtest_corpus_staging_path / "tier-1" / f"doc_{i:05d}.md")
    mock_sync_cycle.side_effect = [
        (_sync_result(2), {"a": _state(1)}),
        (_sync_result(2), {"a": _state(1), "b": _state(2)}),
    ]
    mock_query_phase.return_value = []

    run_load_test(settings=settings)

    cumulative_values = [call.kwargs["cumulative_indexed"] for call in mock_log_batch.call_args_list]
    assert cumulative_values == [2, 4]


def test_report_path_for_uses_a_timestamped_filename(tmp_path):
    from datetime import datetime

    path = _report_path_for(tmp_path, now=datetime(2026, 1, 2, 3, 4, 5))

    assert path == tmp_path / "loadtest-20260102T030405.json"


@patch("agentic_rag.loadtest.runner.log_loadtest_run_complete")
@patch("agentic_rag.loadtest.runner.run_load_test")
def test_main_writes_a_json_report_and_logs_completion(mock_run, mock_log, tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHED_FOLDER_PATH", str(tmp_path / "corpus"))
    monkeypatch.setenv("LOADTEST_RESULTS_PATH", str(tmp_path / "results"))
    mock_run.return_value = LoadTestReport(
        total_indexed=10,
        total_ingestion_failures=0,
        total_indexing_failures=0,
        total_duration_seconds=12.5,
        batch_count=1,
        query_latencies_seconds=[1.0],
    )

    main()

    written = list((tmp_path / "results").glob("loadtest-*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["total_indexed"] == 10
    mock_log.assert_called_once()
