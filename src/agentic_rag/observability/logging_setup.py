from __future__ import annotations

import logging
import sys
from typing import TextIO

_active_handlers: dict[str, logging.Handler] = {}


def configure_json_logging(logger_name: str, *, stream: TextIO | None = None) -> None:
    """Idempotently attach a `StreamHandler` (formatting only
    `%(message)s`, since every caller of this is expected to log a
    single already-JSON-encoded string per call) to the logger named
    `logger_name`, pointed at `stream` (default: the *current*
    `sys.stdout`, resolved when this function runs - not whatever
    `sys.stdout` was at import time; a parameter default bound once at
    import would keep writing to a stale object if anything reassigns
    `sys.stdout` afterwards, e.g. a test framework's output capture).

    Shared by every `observability/*_log.py` module (`request_log.py`,
    `sync_log.py`, `eval_log.py`) rather than each reimplementing its own
    idempotent-handler-attachment logic - one bug fixed here (or one new
    capability added later, like rotating file output) benefits all of
    them instead of needing to be ported to each one by hand.

    Neither the root logger nor uvicorn's own logging config attaches a
    handler to application-level loggers by default - without this, a
    plain `logger.info(...)` call goes nowhere (Python's `lastResort`
    handler only surfaces `WARNING`+ to stderr). `propagate = False`
    stops the same line from also being formatted/emitted a second time
    by whatever the root logger's config happens to be.

    Idempotent *per `logger_name`*: the one handler this function
    previously attached for that specific logger (tracked by direct
    object reference in `_active_handlers`, keyed by name) is removed
    before a new one is added, so calling it more than once for the same
    logger never leaves it writing every line more than once. Each
    logger's handler is tracked independently, so reconfiguring one
    logger never disturbs a different logger's already-attached handler.
    """
    logger = logging.getLogger(logger_name)
    previous = _active_handlers.get(logger_name)
    if previous is not None:
        logger.removeHandler(previous)

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    _active_handlers[logger_name] = handler
    logger.setLevel(logging.INFO)
    logger.propagate = False
