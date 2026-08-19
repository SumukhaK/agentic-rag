from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone

from agentic_rag.observability.sync_log import LOGGER_NAME

# Reuses the same logger sync_log.py / ingestion/scheduler.py's own
# logging.getLogger(__name__) already write through, rather than
# introducing a new logger name with its own configure_*_logging()
# function - backup events fire from inside the same run_sync_loop()
# that logger already covers, and configure_sync_logging() (called once
# at app startup, api/app.py) already attaches a handler to it. See
# sync_log.py's own docstring for the shared-logger reasoning this
# follows.
_logger = logging.getLogger(LOGGER_NAME)


def log_qdrant_backup(*, backup_path: str, duration_seconds: float) -> None:
    """Emit one structured JSON log line for a completed Qdrant storage
    backup (`indexing/backup.py::backup_qdrant_storage()`).
    """
    payload = {
        "event": "qdrant_backup",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backup_path": backup_path,
        "duration_seconds": duration_seconds,
    }
    _logger.info(json.dumps(payload))


def log_qdrant_backup_error(*, error: str, duration_seconds: float) -> None:
    """Emit one structured JSON log line when a backup attempt raises.

    A failed backup is logged, not allowed to crash the sync loop - see
    `ingestion/scheduler.py`'s wiring for why a backup failure must never
    take down index freshness (the sync loop's actual job) along with it.
    Same traceback-as-a-JSON-field approach as
    `sync_log.log_sync_cycle_error()` - see that function's docstring for
    why `exc_info=True` isn't used instead.
    """
    payload = {
        "event": "qdrant_backup_error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "duration_seconds": duration_seconds,
        "traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
    }
    _logger.error(json.dumps(payload))
