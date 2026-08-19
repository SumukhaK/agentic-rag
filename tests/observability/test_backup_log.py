import io
import json
import logging

from agentic_rag.observability.backup_log import log_qdrant_backup, log_qdrant_backup_error
from agentic_rag.observability.sync_log import LOGGER_NAME, configure_sync_logging


def test_log_qdrant_backup_emits_exactly_one_info_record(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_qdrant_backup(backup_path="/data/qdrant_backups/20260101T000000Z", duration_seconds=4.2)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO


def test_log_qdrant_backup_message_is_valid_json_with_expected_fields(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_qdrant_backup(backup_path="/data/qdrant_backups/20260101T000000Z", duration_seconds=4.2)

    payload = json.loads(caplog.records[0].getMessage())
    assert payload["event"] == "qdrant_backup"
    assert payload["backup_path"] == "/data/qdrant_backups/20260101T000000Z"
    assert payload["duration_seconds"] == 4.2
    assert "timestamp" in payload


def test_log_qdrant_backup_reuses_the_scheduler_logger_already_configured_at_startup():
    # backup_log.py deliberately has no configure_*_logging() of its own -
    # it writes through the same logger sync_log.py/scheduler.py already
    # use, which configure_sync_logging() (called once at app startup)
    # already attaches a handler to.
    stream = io.StringIO()
    configure_sync_logging(stream=stream)

    log_qdrant_backup(backup_path="/data/qdrant_backups/x", duration_seconds=1.0)

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "qdrant_backup"


def test_log_qdrant_backup_error_emits_a_distinct_event_with_the_error_message(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_qdrant_backup_error(error="OSError: disk full", duration_seconds=0.8)

    payload = json.loads(caplog.records[0].getMessage())
    assert payload["event"] == "qdrant_backup_error"
    assert payload["error"] == "OSError: disk full"
    assert payload["duration_seconds"] == 0.8


def test_log_qdrant_backup_error_embeds_the_traceback_in_the_json_payload():
    stream = io.StringIO()
    configure_sync_logging(stream=stream)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log_qdrant_backup_error(error="RuntimeError: boom", duration_seconds=0.5)

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert "RuntimeError: boom" in payload["traceback"]
    assert "Traceback" in payload["traceback"]


def test_log_qdrant_backup_error_is_safe_with_no_active_exception():
    stream = io.StringIO()
    configure_sync_logging(stream=stream)

    log_qdrant_backup_error(error="boom", duration_seconds=0.1)

    lines = [line for line in stream.getvalue().splitlines() if line]
    payload = json.loads(lines[0])
    assert payload["traceback"] is None
