import io
import json
import logging

from agentic_rag.observability.sync_log import (
    LOGGER_NAME,
    configure_sync_logging,
    log_sync_cycle,
    log_sync_cycle_error,
)


def test_log_sync_cycle_emits_exactly_one_info_record(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_sync_cycle(
            indexed_count=2,
            deleted_count=1,
            ingestion_failure_count=0,
            indexing_failure_count=0,
            deletion_failure_count=0,
            duration_seconds=3.5,
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO


def test_log_sync_cycle_message_is_valid_json_with_expected_fields(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_sync_cycle(
            indexed_count=2,
            deleted_count=1,
            ingestion_failure_count=1,
            indexing_failure_count=0,
            deletion_failure_count=0,
            duration_seconds=3.5,
        )

    payload = json.loads(caplog.records[0].getMessage())

    assert payload["event"] == "sync_cycle"
    assert payload["indexed_count"] == 2
    assert payload["deleted_count"] == 1
    assert payload["ingestion_failure_count"] == 1
    assert payload["indexing_failure_count"] == 0
    assert payload["deletion_failure_count"] == 0
    assert payload["duration_seconds"] == 3.5
    assert "timestamp" in payload


def test_log_sync_cycle_error_emits_a_distinct_event_with_the_error_message(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_sync_cycle_error(error="RuntimeError: Qdrant unreachable", duration_seconds=1.2)

    payload = json.loads(caplog.records[0].getMessage())
    assert payload["event"] == "sync_cycle_error"
    assert payload["error"] == "RuntimeError: Qdrant unreachable"
    assert payload["duration_seconds"] == 1.2


def test_configure_sync_logging_writes_json_lines_to_the_given_stream():
    stream = io.StringIO()

    configure_sync_logging(stream=stream)
    log_sync_cycle(
        indexed_count=0,
        deleted_count=0,
        ingestion_failure_count=0,
        indexing_failure_count=0,
        deletion_failure_count=0,
        duration_seconds=0.1,
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "sync_cycle"
