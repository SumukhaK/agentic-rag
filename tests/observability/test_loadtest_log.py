import io
import json
import logging

from agentic_rag.observability.loadtest_log import (
    LOGGER_NAME,
    configure_loadtest_logging,
    log_loadtest_batch,
    log_loadtest_run_complete,
)


def _log_loadtest_batch(**overrides):
    defaults = dict(
        batch_index=0,
        batch_size=200,
        indexed_count=200,
        ingestion_failure_count=0,
        indexing_failure_count=0,
        ingestion_failure_paths=[],
        indexing_failure_paths=[],
        duration_seconds=2100.0,
        cumulative_indexed=200,
        cumulative_elapsed_seconds=2100.0,
    )
    defaults.update(overrides)
    log_loadtest_batch(**defaults)


def test_log_loadtest_batch_emits_exactly_one_info_record(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _log_loadtest_batch()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO


def test_log_loadtest_batch_message_is_valid_json_with_expected_fields(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _log_loadtest_batch(
            batch_index=3,
            batch_size=200,
            indexed_count=198,
            ingestion_failure_count=1,
            indexing_failure_count=1,
            ingestion_failure_paths=["employee/doc_00500.md"],
            indexing_failure_paths=["manager/doc_00601.md"],
            duration_seconds=2050.5,
            cumulative_indexed=798,
            cumulative_elapsed_seconds=8300.0,
        )

    payload = json.loads(caplog.records[0].getMessage())

    assert payload["event"] == "loadtest_batch"
    assert payload["batch_index"] == 3
    assert payload["batch_size"] == 200
    assert payload["indexed_count"] == 198
    assert payload["ingestion_failure_count"] == 1
    assert payload["indexing_failure_count"] == 1
    assert payload["ingestion_failure_paths"] == ["employee/doc_00500.md"]
    assert payload["indexing_failure_paths"] == ["manager/doc_00601.md"]
    assert payload["duration_seconds"] == 2050.5
    assert payload["cumulative_indexed"] == 798
    assert payload["cumulative_elapsed_seconds"] == 8300.0
    assert "timestamp" in payload


def test_log_loadtest_batch_includes_failure_paths_not_just_counts():
    # At target scale, which document is persistently failing is what a
    # reader actually needs to grep for - matches sync_log.py's own
    # reasoning for log_sync_cycle().
    stream = io.StringIO()
    configure_loadtest_logging(stream=stream)

    _log_loadtest_batch(
        ingestion_failure_count=1,
        ingestion_failure_paths=["employee/doc_00013.md"],
        indexing_failure_count=1,
        indexing_failure_paths=["employee/doc_00042.md"],
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["ingestion_failure_paths"] == ["employee/doc_00013.md"]
    assert payload["indexing_failure_paths"] == ["employee/doc_00042.md"]


def test_log_loadtest_run_complete_emits_a_distinct_event_with_summary_fields(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_loadtest_run_complete(
            total_indexed=10_000,
            total_ingestion_failures=2,
            total_indexing_failures=1,
            total_duration_seconds=107_000.0,
            query_latencies_seconds=[1.2, 0.9, 1.5],
            report_path="loadtest/results/loadtest-20260101T000000.json",
        )

    payload = json.loads(caplog.records[0].getMessage())
    assert payload["event"] == "loadtest_run_complete"
    assert payload["total_indexed"] == 10_000
    assert payload["total_ingestion_failures"] == 2
    assert payload["total_indexing_failures"] == 1
    assert payload["total_duration_seconds"] == 107_000.0
    assert payload["query_latencies_seconds"] == [1.2, 0.9, 1.5]
    assert payload["report_path"] == "loadtest/results/loadtest-20260101T000000.json"


def test_configure_loadtest_logging_writes_json_lines_to_the_given_stream():
    stream = io.StringIO()

    configure_loadtest_logging(stream=stream)
    _log_loadtest_batch(duration_seconds=0.1)

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "loadtest_batch"
