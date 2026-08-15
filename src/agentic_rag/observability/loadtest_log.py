from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TextIO

from agentic_rag.observability.logging_setup import configure_json_logging

# Matches loadtest/runner.py's own `logging.getLogger(__name__)` - reusing
# the same logger that module already logs through means its
# `logger.info()` call sites and these structured calls end up on the
# same, single configured stream.
LOGGER_NAME = "agentic_rag.loadtest.runner"

_logger = logging.getLogger(LOGGER_NAME)


def configure_loadtest_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.loadtest.runner` logger at `stream` (default:
    the current `sys.stdout`) as one structured JSON line per batch/run
    event. Thin wrapper around `logging_setup.configure_json_logging()` -
    see that function's docstring for the idempotency/stdout-timing
    reasoning shared by every `observability/*_log.py` module.
    """
    configure_json_logging(LOGGER_NAME, stream=stream)


def log_loadtest_batch(
    *,
    batch_index: int,
    batch_size: int,
    indexed_count: int,
    ingestion_failure_count: int,
    indexing_failure_count: int,
    ingestion_failure_paths: list[str],
    indexing_failure_paths: list[str],
    duration_seconds: float,
    cumulative_indexed: int,
    cumulative_elapsed_seconds: float,
) -> None:
    """Emit one structured JSON log line summarizing one completed batch
    of the load test's drip-feed loop (`loadtest/runner.py`).

    A run this long (~30 hours at target scale) needs progress checkable
    by tailing a log, not just a final report at the end - `cumulative_*`
    fields let a reader compute real-time throughput and estimate time
    remaining without replaying every prior line. `ingestion_failure_paths`/
    `indexing_failure_paths` (not just counts) matches
    `sync_log.py::log_sync_cycle()`'s own reasoning: at target scale,
    which document is failing is what a reader actually needs to debug it.
    """
    payload = {
        "event": "loadtest_batch",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_index": batch_index,
        "batch_size": batch_size,
        "indexed_count": indexed_count,
        "ingestion_failure_count": ingestion_failure_count,
        "indexing_failure_count": indexing_failure_count,
        "ingestion_failure_paths": ingestion_failure_paths,
        "indexing_failure_paths": indexing_failure_paths,
        "duration_seconds": duration_seconds,
        "cumulative_indexed": cumulative_indexed,
        "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
    }
    _logger.info(json.dumps(payload))


def log_loadtest_run_complete(
    *,
    total_indexed: int,
    total_ingestion_failures: int,
    total_indexing_failures: int,
    total_duration_seconds: float,
    query_latencies_seconds: list[float],
    report_path: str,
) -> None:
    """Emit one structured JSON log line for the whole load test finishing
    - both the ingestion phase and the post-load query-latency phase
    (`run_load_test()`'s two stages) summarized in a single terminal
    event, so a reader watching the log stream sees an unambiguous "this
    run is done" line rather than inferring completion from the last
    batch line simply stopping.
    """
    payload = {
        "event": "loadtest_run_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_indexed": total_indexed,
        "total_ingestion_failures": total_ingestion_failures,
        "total_indexing_failures": total_indexing_failures,
        "total_duration_seconds": total_duration_seconds,
        "query_latencies_seconds": query_latencies_seconds,
        "report_path": report_path,
    }
    _logger.info(json.dumps(payload))
