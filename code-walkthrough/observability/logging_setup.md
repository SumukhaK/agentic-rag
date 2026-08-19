# `observability/logging_setup.py`

**Purpose:** This file provides one small, shared helper — `configure_json_logging()` — that every other logging module in `observability/` (`request_log.py`, `sync_log.py`, `eval_log.py`, `loadtest_log.py`) calls instead of writing their own setup code. Its job is to take a named Python logger and wire it up so that anything logged through it is printed as a single line of text to a stream (normally the terminal's standard output), without Python's default logging machinery getting in the way or causing duplicate output. Centralizing this in one place means that if a bug is found in how logging gets configured, or a new feature (like writing logs to a rotating file) is needed, it can be fixed or added once here and every module that depends on it benefits automatically, instead of the fix needing to be copy-pasted into five different files.

## Line-by-line walkthrough

### Lines 1-7 — Imports and module-level state
```python
from __future__ import annotations

import logging
import sys
from typing import TextIO

_active_handlers: dict[str, logging.Handler] = {}
```
- `from __future__ import annotations` — turns on a mode where type hints (like `dict[str, logging.Handler]` below) are treated as text rather than evaluated immediately. This lets the code use modern, simpler type-hint syntax even on older Python versions, and slightly speeds up import time since hints aren't built into real objects unless something actually inspects them.
- `import logging` — brings in Python's built-in logging library, which this whole module is a thin wrapper around.
- `import sys` — needed to access `sys.stdout` (the program's standard output stream) as the default place logs get written.
- `from typing import TextIO` — imports the type used to describe "a writable text stream" (like a file or `sys.stdout`), used to type-hint the `stream` parameter below.
- `_active_handlers: dict[str, logging.Handler] = {}` — a module-level dictionary that remembers, for each logger name this function has configured, which specific `Handler` object (the piece of logging machinery that actually writes output somewhere) is currently attached to it. This is the bookkeeping that makes the function "idempotent" (safe to call more than once without piling up duplicate handlers), explained further below.

### Lines 10-18 — Function signature and the "why default to current stdout" reasoning
```python
def configure_json_logging(logger_name: str, *, stream: TextIO | None = None) -> None:
    """Idempotently attach a `StreamHandler` (formatting only
    `%(message)s`, since every caller of this is expected to log a
    single already-JSON-encoded string per call) to the logger named
    `logger_name`, pointed at `stream` (default: the *current*
    `sys.stdout`, resolved when this function runs - not whatever
    `sys.stdout` was at import time; a parameter default bound once at
    import would keep writing to a stale object if anything reassigns
    `sys.stdout` afterwards, e.g. a test framework's output capture).
```
- `def configure_json_logging(logger_name: str, *, stream: TextIO | None = None) -> None:` — defines the public function. It takes the name of the logger to configure (a plain string, e.g. `"agentic_rag.query"`) and an optional `stream` to write to. The `*` forces `stream` to be passed by keyword (`stream=...`), not positionally, which makes call sites self-documenting. Defaulting `stream` to `None` rather than `sys.stdout` directly is a deliberate choice: `sys.stdout` is evaluated fresh every time the function body runs, not once when the module is first imported. If it were baked in as the default value at import time, and something later replaced `sys.stdout` (as test frameworks commonly do to capture output), this function would keep writing to the old, now-disconnected stream.
- The docstring explains that the caller is expected to log one already-JSON-encoded string per call — meaning the `%(message)s` formatter (set up below) doesn't need to add timestamps, log levels, or any other decoration, because the JSON string itself already carries all the structured information.

### Lines 20-24 — Why this logic lives in a shared module
```python
    Shared by every `observability/*_log.py` module (`request_log.py`,
    `sync_log.py`, `eval_log.py`) rather than each reimplementing its own
    idempotent-handler-attachment logic - one bug fixed here (or one new
    capability added later, like rotating file output) benefits all of
    them instead of needing to be ported to each one by hand.
```
- This part of the docstring makes explicit the design rationale: rather than every `*_log.py` file duplicating the same handler-setup code, they all import and call this one function. That means a future improvement (e.g. supporting log rotation to files) only needs to be written once.

### Lines 26-31 — Why a handler must be attached at all, and why `propagate = False`
```python
    Neither the root logger nor uvicorn's own logging config attaches a
    handler to application-level loggers by default - without this, a
    plain `logger.info(...)` call goes nowhere (Python's `lastResort`
    handler only surfaces `WARNING`+ to stderr). `propagate = False`
    stops the same line from also being formatted/emitted a second time
    by whatever the root logger's config happens to be.
```
- This explains two things a newcomer might not know about Python's `logging` module: first, that simply calling `logger.info(...)` on a fresh logger normally produces no visible output at all (below `WARNING` severity), because nothing is listening for it by default — hence the need to explicitly attach a handler. Second, that loggers by default "bubble up" (propagate) their messages to the root logger too, which could cause the same JSON line to be printed twice if the root logger also has its own handler (as `uvicorn`, the web server this project runs under, sets up for itself). Setting `propagate = False` (line 52) prevents that double-print.

### Lines 33-40 — Idempotency guarantee
```python
    Idempotent *per `logger_name`*: the one handler this function
    previously attached for that specific logger (tracked by direct
    object reference in `_active_handlers`, keyed by name) is removed
    before a new one is added, so calling it more than once for the same
    logger never leaves it writing every line more than once. Each
    logger's handler is tracked independently, so reconfiguring one
    logger never disturbs a different logger's already-attached handler.
    """
```
- This is the core promise of the function: it can safely be called multiple times for the same logger (for example, if a test suite calls the configuration function before every test) without ending up with multiple handlers stacked on top of each other, each printing the same line again. It achieves this using the `_active_handlers` dictionary from line 7, keyed by logger name, so different loggers' setups never interfere with one another.

### Lines 41-44 — Look up the logger and remove any previous handler
```python
    logger = logging.getLogger(logger_name)
    previous = _active_handlers.get(logger_name)
    if previous is not None:
        logger.removeHandler(previous)
```
- `logger = logging.getLogger(logger_name)` — asks Python's logging system for the (singleton) logger object with this name, creating it if it doesn't already exist. Because `logging.getLogger` always returns the same object for the same name, this is safe to call repeatedly.
- `previous = _active_handlers.get(logger_name)` — checks whether this function previously attached a handler to this exact logger name, by looking it up in the module-level tracking dictionary.
- `if previous is not None: logger.removeHandler(previous)` — if a handler was attached before, it's removed first. This is what actually implements the idempotency described above — without this step, calling the function twice would leave two handlers both writing the same message.

### Lines 46-47 — Build the new handler
```python
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
```
- `handler = logging.StreamHandler(stream if stream is not None else sys.stdout)` — creates a new `StreamHandler`, the logging building block that writes formatted log records to a text stream. If the caller passed an explicit `stream`, it's used; otherwise `sys.stdout` is read right now (not at some earlier time), matching the "current stdout" reasoning from the docstring.
- `handler.setFormatter(logging.Formatter("%(message)s"))` — configures the handler to output *only* the raw message text, with no extra prefix (no timestamp, level name, logger name, etc. added by the formatter). This matters because every caller of this function already logs a complete, pre-built JSON string as the message — adding any decoration here would break the "one clean JSON object per line" contract that downstream tools (e.g. log parsers) rely on.

### Lines 49-52 — Attach the handler and finish configuring the logger
```python
    logger.addHandler(handler)
    _active_handlers[logger_name] = handler
    logger.setLevel(logging.INFO)
    logger.propagate = False
```
- `logger.addHandler(handler)` — actually attaches the newly built handler to the logger, so from this point on, messages logged through it get written to the stream.
- `_active_handlers[logger_name] = handler` — records this handler in the tracking dictionary so a future call to `configure_json_logging()` for the same logger name knows to remove it first (see lines 42-44).
- `logger.setLevel(logging.INFO)` — sets the minimum severity this logger will process to `INFO`. Without this, a freshly created logger defaults to a level that could silently drop `INFO`-level calls (like the `logger.info(json.dumps(payload))` calls used throughout the other modules).
- `logger.propagate = False` — stops messages logged through this logger from also being passed up to the root logger's own handlers, preventing the duplicate-output problem described in the docstring above.
