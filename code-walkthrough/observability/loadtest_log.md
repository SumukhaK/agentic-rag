# `observability/loadtest_log.py`

**Purpose:** This file produces structured (JSON, one event per line) log output for the project's load-testing tool (`loadtest/runner.py`), which simulates ingesting and querying a very large number of documents (tens of thousands) over a very long run (roughly 30 hours at full target scale) to see how the system behaves under sustained load. Because a run this long can't be watched end-to-end by a human sitting at a terminal, this module's job is to make the *log stream itself* a useful, real-time progress report — someone can `tail` the log file at any point and understand how much work has been done, how much has failed, and roughly how fast things are going, without waiting for a final summary. It reuses the shared logger-configuration helper from `logging_setup.py` rather than reimplementing that logic.

## Line-by-line walkthrough

### Lines 1-8 — Imports
```python
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TextIO

from agentic_rag.observability.logging_setup import configure_json_logging
```
- `from __future__ import annotations` — enables the modern, lazily-evaluated type hint syntax used later in the file (e.g. `TextIO | None`), and avoids needless runtime overhead from evaluating those hints immediately.
- `import json` — used to turn the payload dictionaries built in this file into JSON-formatted strings before they're logged.
- `import logging` — Python's standard logging library, used to obtain the `Logger` object this module writes through.
- `from datetime import datetime, timezone` — used to stamp each log line with the current UTC time.
- `from typing import TextIO` — the type used for the optional `stream` parameter (a writable text stream).
- `from agentic_rag.observability.logging_setup import configure_json_logging` — imports the one shared function responsible for actually attaching and configuring a logging handler; this module only supplies its own logger name and payload shape.

### Lines 10-16 — Logger name and module-level logger
```python
# Matches loadtest/runner.py's own `logging.getLogger(__name__)` - reusing
# the same logger that module already logs through means its
# `logger.info()` call sites and these structured calls end up on the
# same, single configured stream.
LOGGER_NAME = "agentic_rag.loadtest.runner"

_logger = logging.getLogger(LOGGER_NAME)
```
- The comment explains that this constant is chosen to exactly match the logger name that `loadtest/runner.py` itself would get by calling `logging.getLogger(__name__)` inside that module. Because Python's `logging.getLogger()` always returns the same object for the same name, this means any plain `logger.info(...)` calls already present in `loadtest/runner.py`, and the structured JSON calls made from this file, both end up going through the exact same configured handler and output stream — nothing gets split across two different destinations.
- `LOGGER_NAME = "agentic_rag.loadtest.runner"` — the constant holding that shared name.
- `_logger = logging.getLogger(LOGGER_NAME)` — fetches that logger object once at import time and stores it for reuse by the logging functions below.

### Lines 19-26 — `configure_loadtest_logging()`
```python
def configure_loadtest_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.loadtest.runner` logger at `stream` (default:
    the current `sys.stdout`) as one structured JSON line per batch/run
    event. Thin wrapper around `logging_setup.configure_json_logging()` -
    see that function's docstring for the idempotency/stdout-timing
    reasoning shared by every `observability/*_log.py` module.
    """
    configure_json_logging(LOGGER_NAME, stream=stream)
```
- `def configure_loadtest_logging(*, stream: TextIO | None = None) -> None:` — the setup function a caller runs once before the load test starts logging, to wire up the destination stream. `stream` is keyword-only and defaults to `None`, letting `configure_json_logging` resolve "current `sys.stdout`" at call time rather than baking in a possibly-stale reference.
- The docstring points readers to `logging_setup.py`'s own docstring rather than repeating the idempotency (safe to call more than once) and stdout-timing reasoning, since that logic is identical across all four `*_log.py` modules.
- `configure_json_logging(LOGGER_NAME, stream=stream)` — the whole function body: delegate to the shared helper with this module's specific logger name.

### Lines 29-41 — `log_loadtest_batch()` signature
```python
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
```
- All parameters are keyword-only, forcing every call site to name them, which prevents mistakes given how many `int`/`float` parameters of similar shape this function has.
- `batch_index` / `batch_size` — which batch (chunk of documents processed together) this log line describes, and how many documents were in it.
- `indexed_count` — how many documents from this batch were successfully indexed.
- `ingestion_failure_count` / `indexing_failure_count` — how many documents failed during the two distinct phases (ingestion — reading/parsing a source document — versus indexing — storing it into the retrieval system) this batch went through.
- `ingestion_failure_paths` / `indexing_failure_paths` — the actual file paths that failed in each phase, not just counts (explained further below).
- `duration_seconds` — how long this one batch took to process.
- `cumulative_indexed` / `cumulative_elapsed_seconds` — running totals across every batch processed so far in this load test run, not just this batch.

### Lines 42-52 — Docstring: why cumulative fields and failure paths exist
```python
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
```
- This explains the two central design decisions in this function's payload. First, because the whole load test can run for around 30 hours, someone needs to be able to check progress live by watching the log stream, rather than only finding out how things went once the entire run finishes. Including `cumulative_indexed` and `cumulative_elapsed_seconds` (running totals, not just this batch's numbers) in every batch's log line lets a reader compute throughput (documents per second) and estimate remaining time using only the single most recent line, without needing to add up every prior line themselves.
- Second, it explains why full failure *paths* are logged rather than just failure *counts*: at the scale this load test operates at, when something is failing repeatedly, knowing exactly which document is the problem is what actually lets someone debug it — a bare count like "3 failures" gives no way to investigate. This mirrors the same reasoning used in `sync_log.py`'s `log_sync_cycle()` function.

### Lines 53-67 — Building and emitting the batch payload
```python
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
```
- `payload = {...}` — assembles a dictionary with a fixed `"event": "loadtest_batch"` tag (so this specific kind of log line can be distinguished from other event types in the same stream), a UTC `"timestamp"` computed the same way as in the other `*_log.py` modules, and then every parameter passed into the function copied straight across as a matching key.
- `_logger.info(json.dumps(payload))` — serializes the dictionary to a JSON string and writes it as one line through the configured logger, exactly as in the other observability modules.

### Lines 70-77 — `log_loadtest_run_complete()` signature
```python
def log_loadtest_run_complete(
    *,
    total_indexed: int,
    total_ingestion_failures: int,
    total_indexing_failures: int,
    total_duration_seconds: float,
    query_latencies_seconds: list[float],
    report_path: str,
) -> None:
```
- Again keyword-only parameters. This function is called exactly once, at the very end of the entire load test run (not per-batch), to summarize the whole thing.
- `total_indexed` / `total_ingestion_failures` / `total_indexing_failures` — grand totals across every batch of the ingestion phase.
- `total_duration_seconds` — how long the whole run took, start to finish.
- `query_latencies_seconds` — a list of how long each individual test query took to answer, from the load test's separate post-ingestion querying phase.
- `report_path` — where the full, detailed report for this run was written to disk.

### Lines 78-85 — Docstring: why a distinct "run complete" event exists
```python
    """Emit one structured JSON log line for the whole load test finishing
    - both the ingestion phase and the post-load query-latency phase
    (`run_load_test()`'s two stages) summarized in a single terminal
    event, so a reader watching the log stream sees an unambiguous "this
    run is done" line rather than inferring completion from the last
    batch line simply stopping.
    """
```
- Explains why this function exists separately from `log_loadtest_batch()`: without an explicit "the run is complete" event, someone watching the log stream would have no reliable way to tell the difference between "the load test finished successfully" and "the load test process crashed or hung partway through" — both would just look like the batch lines stopping. Emitting one clear terminal event removes that ambiguity, and it covers both of the load test's two stages (bulk ingestion, then measuring query latency afterward) in a single summary line.

### Lines 86-96 — Building and emitting the completion payload
```python
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
```
- Same pattern as before: a dictionary tagged with a distinct `"event": "loadtest_run_complete"` value (so it's easy to filter for exactly the "run finished" lines among many batch lines), a UTC timestamp, and every function parameter copied straight into the payload — including the raw list of per-query latencies, so a reader (or an analysis script) can compute percentiles, averages, or spot outliers directly from the log line without needing to open the separate report file.
- `_logger.info(json.dumps(payload))` — serializes and emits the line the same way as every other logging call in this file and its sibling modules.
