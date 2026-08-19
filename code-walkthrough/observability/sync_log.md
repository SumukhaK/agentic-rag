# `observability/sync_log.py`

**Purpose:** This file produces structured (JSON, one self-contained event per line) log output for the background data-synchronization process (`ingestion/scheduler.py`), which periodically scans the document corpus for additions, changes, and deletions and updates the retrieval system to match. Its job is to make each completed "sync cycle" (one full pass of that scan-and-update process) show up as a single, clear summary log line — including, critically, which specific documents failed if anything went wrong, since with potentially 10,000+ documents in the corpus a bare failure count would give no way to actually track down the problem. It also handles the rarer case where an entire sync cycle crashes outright. Like its sibling modules, it relies on the shared handler-configuration logic in `logging_setup.py` instead of duplicating that setup code.

## Line-by-line walkthrough

### Lines 1-10 — Imports
```python
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import TextIO

from agentic_rag.observability.logging_setup import configure_json_logging
```
- `from __future__ import annotations` — enables the modern, lazily-evaluated type hint syntax used elsewhere in the file, without evaluating hints at runtime unless something actually inspects them.
- `import json` — used to convert the payload dictionaries built in this file into JSON strings before logging them.
- `import logging` — Python's standard logging library, used to obtain the `Logger` object this module writes through.
- `import sys` — used specifically to call `sys.exc_info()`, which reports whether an exception is currently being handled (see `log_sync_cycle_error()` below).
- `import traceback` — used to format the currently active exception's traceback (the call-stack trail showing where an error occurred) as a string, so it can be embedded directly in the JSON payload.
- `from datetime import datetime, timezone` — used to stamp each log line with the current UTC time.
- `from typing import TextIO` — the type used for the optional `stream` parameter.
- `from agentic_rag.observability.logging_setup import configure_json_logging` — imports the shared helper responsible for actually attaching and configuring the logging handler.

### Lines 12-19 — Logger name and module-level logger
```python
# Matches ingestion/scheduler.py's own `logging.getLogger(__name__)` -
# reusing the same logger that module already logs through (rather than a
# separate name this module would have to introduce) means the existing
# `logger.info()`/`logger.exception()` call sites and these structured
# calls end up on the same, single configured stream.
LOGGER_NAME = "agentic_rag.ingestion.scheduler"

_logger = logging.getLogger(LOGGER_NAME)
```
- The comment explains that this constant is deliberately chosen to match the logger name `ingestion/scheduler.py` would get on its own via `logging.getLogger(__name__)`. Since Python's `logging.getLogger()` returns the same object for a given name no matter where it's requested from, this means any ordinary `logger.info()` or `logger.exception()` calls already written inside `ingestion/scheduler.py`, and the structured JSON-emitting calls defined in this file, both end up flowing through the exact same configured handler and output stream.
- `LOGGER_NAME = "agentic_rag.ingestion.scheduler"` — the constant holding that shared logger name.
- `_logger = logging.getLogger(LOGGER_NAME)` — fetches that logger object once at import time, stored for reuse by the functions below.

### Lines 22-29 — `configure_sync_logging()`
```python
def configure_sync_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.ingestion.scheduler` logger at `stream`
    (default: the current `sys.stdout`) as one structured JSON line per
    sync cycle. Thin wrapper around `logging_setup.configure_json_logging()`
    - see that function's docstring for the idempotency/stdout-timing
    reasoning shared by every `observability/*_log.py` module.
    """
    configure_json_logging(LOGGER_NAME, stream=stream)
```
- `def configure_sync_logging(*, stream: TextIO | None = None) -> None:` — the setup function called once (typically when the sync scheduler starts) to wire this logger to an output stream. `stream` is keyword-only, defaulting to `None` so the "current `sys.stdout`" resolution happens inside `configure_json_logging` itself, avoiding a stale reference.
- The docstring points to `logging_setup.py` for the shared reasoning about idempotency (safe to call more than once without duplicate output) and why stdout is resolved lazily.
- `configure_json_logging(LOGGER_NAME, stream=stream)` — the entire body: delegates to the shared helper with this module's specific logger name.

### Lines 32-43 — `log_sync_cycle()` signature
```python
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
```
- Keyword-only parameters throughout, so call sites must name every argument.
- `indexed_count` / `deleted_count` — how many documents were successfully added/updated in the retrieval index, and how many were removed, during this cycle.
- `ingestion_failure_count`, `indexing_failure_count`, `deletion_failure_count` — how many documents failed at each of the three distinct stages a sync cycle can involve: ingestion (reading/parsing a source document), indexing (storing it in the retrieval system), and deletion (removing a document that no longer exists in the source).
- `ingestion_failure_paths`, `indexing_failure_paths`, `deletion_failure_paths` — the actual file paths that failed at each of those stages, not just the counts (see the docstring below for why).
- `duration_seconds` — how long this whole sync cycle took.

### Lines 44-60 — Docstring: counts for successes, paths for failures, and why
```python
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
```
- Explains an asymmetric design choice in what gets logged in full detail versus just as a number. For *successful* work (indexed/deleted documents), only counts are logged — at the scale this system targets (10,000+ documents), a healthy cycle could touch enough documents that listing every single path would make the log line huge and unreadable, and for a cycle that's working correctly, the number of changes is really all a reader needs to know.
- For *failures*, the opposite choice is made: full paths are logged, not just counts. The reasoning given is that failures are expected to be rare (because each document/deletion is processed independently and isolated from the others — a single bad document shouldn't be able to derail the whole cycle), and when a failure does happen, knowing exactly *which* document failed is the entire point of the log line from a debugging perspective. The docstring draws a direct parallel to `request_log.py`'s `cited_paths` field, which follows the same "a reader debugging something needs to see what was specifically flagged" logic. It also gives a concrete illustration: without logging the actual failing path, a document that fails every single cycle would just show up as `"indexing_failure_count": 1` forever, giving no clue which of potentially 10,000 files is the culprit — someone would have to reproduce the entire run locally just to find out.

### Lines 61-73 — Building and emitting the sync-cycle payload
```python
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
```
- `payload = {...}` — assembles the dictionary that becomes this event's JSON log line, tagged `"event": "sync_cycle"` so it's distinguishable from other event types (including the error variant defined further down this same file), with a UTC `"timestamp"` computed the same way as in every other `observability/*_log.py` module, followed by every function parameter copied straight across.
- `_logger.info(json.dumps(payload))` — serializes the dictionary to a JSON string and writes it through the configured logger, at `INFO` severity, since this represents a normal (even if partially failed) completed cycle.

### Lines 77-101 — `log_sync_cycle_error()` signature and docstring
```python
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
```
- `def log_sync_cycle_error(*, error: str, duration_seconds: float) -> None:` — a separate function, called only when an entire sync cycle raises an exception rather than completing (even partially) normally. `error` is a description of what went wrong; `duration_seconds` is how long the cycle ran before it failed. Both parameters are keyword-only.
- The first docstring paragraph explains this is expected to be a rare event: since every individual document's ingestion/indexing/deletion step is already isolated from the others inside `run_sync_cycle()` (a single bad document can't crash the whole cycle), this function firing at all means something genuinely unexpected happened — a failure mode the per-document isolation wasn't designed to catch.
- The second paragraph explains why a distinct event name (`"sync_cycle_error"`) is used instead of reusing the normal `"sync_cycle"` event with every count set to zero: a cycle that crashed before finishing has no real counts to report — it's not that zero documents happened to change, it's that the process never got far enough to know. Logging it with all zeros would be actively misleading, making a crashed run look identical to a run that legitimately had nothing to do.
- The third paragraph explains a specific, deliberate choice about *how* the traceback (the detailed trail showing where in the code an exception occurred) is captured. Python's `logging` module has a built-in way to attach this information (`exc_info=True`), but using it would print the traceback as extra plain-text lines appended after the main log message — breaking the "each log line is one complete, independently-parseable JSON object" rule every other call in this whole module family follows. It would also produce a nonsensical fake traceback (literally reading `"NoneType: None"`) if this function were ever accidentally called when there's no exception actually active. Instead, this function manually captures the traceback into the JSON payload itself, using an approach (`traceback.format_exc()`, guarded by checking `sys.exc_info()`) that safely returns `None` when there's genuinely no exception in flight, and the real formatted traceback text when there is — keeping the same diagnostic usefulness the codebase previously got from a plain `logger.exception(...)` call, but without breaking the one-JSON-line-per-event contract.

### Lines 102-109 — Building and emitting the error payload
```python
    payload = {
        "event": "sync_cycle_error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "duration_seconds": duration_seconds,
        "traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
    }
    _logger.error(json.dumps(payload))
```
- `payload = {...}` — builds the dictionary for this error event: tagged `"event": "sync_cycle_error"` (distinct from the normal `"sync_cycle"` tag, as explained above), with the same UTC timestamp convention, the `error` description and `duration_seconds` passed straight through from the function's parameters.
- `"traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None` — this is the safe traceback-capture logic described in the docstring: `sys.exc_info()[0]` reports the *type* of the exception currently being handled (or `None` if there isn't one). If there genuinely is an active exception, `traceback.format_exc()` is called to produce the full formatted traceback string; otherwise the field is explicitly set to `None` rather than risking a misleading placeholder.
- `_logger.error(json.dumps(payload))` — serializes the payload to JSON and logs it, this time at `ERROR` severity (rather than `INFO`, as used by every other logging call across this module family), reflecting that this represents a genuine failure rather than a normal completed cycle.
