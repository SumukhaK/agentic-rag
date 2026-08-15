from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import TextIO

from agentic_rag.observability.logging_setup import configure_json_logging

# Matches ingestion/scheduler.py's own `logging.getLogger(__name__)` -
# reusing the same logger that module already logs through (rather than a
# separate name this module would have to introduce) means the existing
# `logger.info()`/`logger.exception()` call sites and these structured
# calls end up on the same, single configured stream.
LOGGER_NAME = "agentic_rag.ingestion.scheduler"

_logger = logging.getLogger(LOGGER_NAME)


def configure_sync_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.ingestion.scheduler` logger at `stream`
    (default: the current `sys.stdout`) as one structured JSON line per
    sync cycle. Thin wrapper around `logging_setup.configure_json_logging()`
    - see that function's docstring for the idempotency/stdout-timing
    reasoning shared by every `observability/*_log.py` module.
    """
    configure_json_logging(LOGGER_NAME, stream=stream)


def log_sync_cycle(
    *,
    indexed_count: int,
    deleted_count: int,
    ingestion_failure_count: int,
    indexing_failure_count: int,
    deletion_failure_count: int,
    ingestion_failure_paths: list[str],
    indexing_failure_paths: list[str],
    deletion_failure_paths: list[str],
    duration_seconds: float,
) -> None:
    """Emit one structured JSON log line summarizing one completed
    `run_sync_cycle()` call (`ingestion/scheduler.py`).

    Successfully indexed/deleted paths are counts only, not the actual
    paths - at target scale (10,000+ documents) a cycle could touch a
    number of paths large enough to make a single log line unwieldy, and
    "how many changed" is what matters for a healthy cycle. Failure
    paths are the opposite: failures are rare by design (every per-
    document/deletion step is already isolated - see `run_sync_cycle()`'s
    own docstring), and *which* document failed is exactly what a reader
    needs to actually debug it - the same "a reader debugging why needs
    to see what it flagged" reasoning `request_log.py`'s `cited_paths`
    already applies to a flagged answer. Without this, a document
    silently failing every cycle would show only `indexing_failure_
    count: 1` forever, with no way to identify which of 10,000 files it
    is short of reproducing the run locally.
    """
    payload = {
        "event": "sync_cycle",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "indexed_count": indexed_count,
        "deleted_count": deleted_count,
        "ingestion_failure_count": ingestion_failure_count,
        "indexing_failure_count": indexing_failure_count,
        "deletion_failure_count": deletion_failure_count,
        "ingestion_failure_paths": ingestion_failure_paths,
        "indexing_failure_paths": indexing_failure_paths,
        "deletion_failure_paths": deletion_failure_paths,
        "duration_seconds": duration_seconds,
    }
    _logger.info(json.dumps(payload))


def log_sync_cycle_error(*, error: str, duration_seconds: float) -> None:
    """Emit one structured JSON log line for a whole cycle raising -
    rare, since every per-document/deletion failure is already isolated
    inside `run_sync_cycle()` itself (see that function's own docstring),
    so this only fires for a genuinely unanticipated failure mode.

    A distinct `event` (`sync_cycle_error`, not `sync_cycle`) rather than
    a `sync_cycle` line with all-zero counts - a cycle that never ran to
    completion produced no real counts to report at all, and collapsing
    that into zeros would misrepresent "the cycle didn't run" as "the
    cycle ran and did nothing."

    The traceback of the current exception (if any) is embedded as a
    `traceback` field in the JSON payload itself, rather than relying on
    `logging`'s own `exc_info=True` handling - that appends the
    formatted traceback as extra, non-JSON lines *after* the message,
    breaking the "one JSON line per event" contract every other call in
    this module family keeps, and silently emits a bogus "NoneType:
    None" traceback if this is ever called with no exception actually in
    flight. `traceback.format_exc()` (via `sys.exc_info()`) is safe to
    call either way: `None` when nothing is active, the real formatted
    traceback when there is - preserving the diagnostic value the
    previous plain `logger.exception("sync cycle failed")` call
    provided, without breaking parseability.
    """
    payload = {
        "event": "sync_cycle_error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "duration_seconds": duration_seconds,
        "traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
    }
    _logger.error(json.dumps(payload))
