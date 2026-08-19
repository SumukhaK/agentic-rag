# `observability/backup_log.py`

**Purpose:** This file writes one line of structured (machine-readable) log output every time a Qdrant storage backup succeeds or fails, so an operator can see backups actually happening (or find out quickly if they've silently stopped working) without having to dig through unstructured text logs. It deliberately reuses the exact same logger that `ingestion/scheduler.py`'s other logging already writes through, instead of setting up a new one, since backup events happen from inside that same background loop.

## Line-by-line walkthrough

### Lines 1-10 — Imports and reusing the scheduler's logger
```python
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone

from agentic_rag.observability.sync_log import LOGGER_NAME

_logger = logging.getLogger(LOGGER_NAME)
```
- `from __future__ import annotations` — same forward-compatibility convenience as other files in this codebase.
- `import json` — used to turn a Python dictionary into a JSON text string before logging it, so each log line is valid, parseable JSON.
- `import logging` — Python's standard logging library.
- `import sys` — used later to check whether an exception is currently "in flight" (i.e., this code is running inside an `except` block).
- `import traceback` — used to capture the full text of an exception's traceback (the stack of function calls that led to the error), so it can be embedded in the log line.
- `from datetime import datetime, timezone` — used to stamp each log line with the current UTC time.
- `from agentic_rag.observability.sync_log import LOGGER_NAME` — rather than inventing its own logger name, this file imports the *exact same* logger name `sync_log.py` (and `ingestion/scheduler.py` itself) already use. This means whatever configured that logger once (`configure_sync_logging()`, called at app startup) already covers this file's log calls too - there's no separate setup step needed here.
- `_logger = logging.getLogger(LOGGER_NAME)` — gets a handle to that shared logger object, stored in a module-level variable (the leading underscore signals it's private to this file) so every function below can reuse it.

### Lines 13-22 — `log_qdrant_backup`: logging a successful backup
```python
def log_qdrant_backup(*, backup_path: str, duration_seconds: float) -> None:
    """..."""
    payload = {
        "event": "qdrant_backup",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backup_path": backup_path,
        "duration_seconds": duration_seconds,
    }
    _logger.info(json.dumps(payload))
```
- `def log_qdrant_backup(*, backup_path: str, duration_seconds: float) -> None:` — takes the location the backup was written to and how long it took, both required to be passed by name (the `*`) rather than by position, which makes call sites self-documenting and prevents accidentally swapping the two arguments.
- `payload = {...}` — builds a plain dictionary describing this one event: a fixed `"event"` label (`"qdrant_backup"`) that a log-reading tool can filter on, the current timestamp, the backup's location, and how long it took.
- `_logger.info(json.dumps(payload))` — converts the dictionary to a JSON string and writes it as one log line at the "info" (routine, not a problem) severity level.

### Lines 25-43 — `log_qdrant_backup_error`: logging a failed backup
```python
def log_qdrant_backup_error(*, error: str, duration_seconds: float) -> None:
    """..."""
    payload = {
        "event": "qdrant_backup_error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "duration_seconds": duration_seconds,
        "traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
    }
    _logger.error(json.dumps(payload))
```
- `def log_qdrant_backup_error(*, error: str, duration_seconds: float) -> None:` — takes a short description of what went wrong and how long the attempt ran before failing.
- The docstring explains this is called from inside `ingestion/scheduler.py`'s loop when a backup attempt raises an exception - and that the failure is logged, not allowed to crash the background sync loop, since a broken backup must never stop the more important job of keeping the search index itself up to date.
- `payload = {...}` — same shape as the success case, but with a different fixed `"event"` label (`"qdrant_backup_error"`, so log tooling can tell successes and failures apart without parsing the message text) and two extra fields: the error description, and...
- `"traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None` — if this function is being called from inside a real `except` block (i.e., there's an actual exception currently active), `traceback.format_exc()` captures its full formatted stack trace as text, which is invaluable for debugging. `sys.exc_info()[0] is not None` checks whether there's actually an exception in flight first - calling `format_exc()` when there isn't one would otherwise produce a misleading "no traceback" placeholder instead of a clean `None`. Embedding the traceback as a field *inside* the JSON payload (rather than letting Python's logging library print it separately, after the message) keeps every single log line valid, self-contained JSON - a log-reading tool never has to worry about a stray, non-JSON traceback dump breaking its parser.
- `_logger.error(json.dumps(payload))` — writes the JSON string at the "error" severity level, higher than the success case's "info" level, so it's easier to filter for and more likely to trigger an alert in a real monitoring setup.
