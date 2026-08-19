import io
import json
import logging

from agentic_rag.observability.sync_log import (
    LOGGER_NAME,
    configure_sync_logging,
    log_sync_cycle,
    log_sync_cycle_error,
)


def _log_sync_cycle(**overrides):
    defaults = dict(
        indexed_count=2,
        deleted_count=1,
        ingestion_failure_count=0,
        indexing_failure_count=0,
        deletion_failure_count=0,
        ingestion_failure_paths=[],
        indexing_failure_paths=[],
        deletion_failure_paths=[],
        duration_seconds=3.5,
    )
    defaults.update(overrides)
    log_sync_cycle(**defaults)


def test_log_sync_cycle_emits_exactly_one_info_record(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _log_sync_cycle()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO


def test_log_sync_cycle_message_is_valid_json_with_expected_fields(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _log_sync_cycle(
            indexed_count=2,
            deleted_count=1,
            ingestion_failure_count=1,
            indexing_failure_count=0,
            deletion_failure_count=0,
            ingestion_failure_paths=["employee/bad.md"],
            duration_seconds=3.5,
        )

    payload = json.loads(caplog.records[0].getMessage())

    assert payload["event"] == "sync_cycle"
    assert payload["indexed_count"] == 2
    assert payload["deleted_count"] == 1
    assert payload["ingestion_failure_count"] == 1
    assert payload["indexing_failure_count"] == 0
    assert payload["deletion_failure_count"] == 0
    assert payload["ingestion_failure_paths"] == ["employee/bad.md"]
    assert payload["indexing_failure_paths"] == []
    assert payload["deletion_failure_paths"] == []
    assert payload["duration_seconds"] == 3.5
    assert "timestamp" in payload


def test_log_sync_cycle_includes_failure_paths_not_just_counts():
    # Counts alone can't tell an on-call engineer *which* document is
    # persistently failing to index at target scale - a count-only line
    # would show "indexing_failure_count: 1" forever with nothing to
    # grep for.
    stream = io.StringIO()
    configure_sync_logging(stream=stream)

    _log_sync_cycle(
        indexing_failure_count=1,
        indexing_failure_paths=["manager/broken.md"],
        deletion_failure_count=1,
        deletion_failure_paths=["employee/gone.md"],
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["indexing_failure_paths"] == ["manager/broken.md"]
    assert payload["deletion_failure_paths"] == ["employee/gone.md"]


def test_log_sync_cycle_error_emits_a_distinct_event_with_the_error_message(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_sync_cycle_error(error="RuntimeError: Qdrant unreachable", duration_seconds=1.2)

    payload = json.loads(caplog.records[0].getMessage())
    assert payload["event"] == "sync_cycle_error"
    assert payload["error"] == "RuntimeError: Qdrant unreachable"
    assert payload["duration_seconds"] == 1.2


def test_log_sync_cycle_error_embeds_the_traceback_in_the_json_payload():
    # Embedding the traceback as a JSON field (rather than relying on
    # logging's own exc_info=True, which appends it as extra non-JSON
    # lines after the message) keeps this a single, always-parseable
    # JSON line - the same "one JSON line per event" contract every
    # other observability call in this codebase keeps.
    stream = io.StringIO()
    configure_sync_logging(stream=stream)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log_sync_cycle_error(error="RuntimeError: boom", duration_seconds=0.5)

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert "RuntimeError: boom" in payload["traceback"]
    assert "Traceback" in payload["traceback"]


def test_log_sync_cycle_error_is_safe_with_no_active_exception():
    # This must only ever be called from inside an except block in
    # production, but must not corrupt the log stream if it isn't -
    # exc_info=True with no active exception used to append a bogus
    # "NoneType: None" line.
    stream = io.StringIO()
    configure_sync_logging(stream=stream)

    log_sync_cycle_error(error="boom", duration_seconds=0.1)

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["traceback"] is None


def test_configure_sync_logging_writes_json_lines_to_the_given_stream():
    stream = io.StringIO()

    configure_sync_logging(stream=stream)
    _log_sync_cycle(duration_seconds=0.1)

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "sync_cycle"
