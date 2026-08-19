# `observability/eval_log.py`

**Purpose:** This file is responsible for producing one line of structured (machine-readable, JSON-formatted) logging output every time the evaluation runner (`evaluation/runner.py`) finishes a full evaluation run of the assistant's answer quality. Structured logging means each log line is a JSON object with a fixed, predictable set of fields rather than a free-form sentence, so tools (or humans skimming logs) can reliably parse and aggregate metrics like retrieval precision or hallucination rate across many runs over time. It builds directly on top of the shared `logging_setup.py` helper described elsewhere in this walkthrough, rather than reinventing logger configuration itself.

## Line-by-line walkthrough

### Lines 1-9 — Imports
```python
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from agentic_rag.observability.logging_setup import configure_json_logging
```
- `from __future__ import annotations` — enables modern, lazily-evaluated type hint syntax (like `float | None` below) regardless of Python version, and avoids the runtime cost of building hint objects that are never inspected.
- `import json` — used to serialize the Python dictionary built later in the file into a JSON-formatted string before logging it.
- `import logging` — Python's standard logging library; this module gets a `Logger` object from it and calls `.info(...)` on it.
- `from datetime import datetime, timezone` — used to generate a timezone-aware "now" timestamp for each log line.
- `from pathlib import Path` — used to pull a short identifier out of a full file path (see `report_id` below).
- `from typing import TextIO` — the type used to describe "a writable text stream," used for the optional `stream` parameter.
- `from agentic_rag.observability.logging_setup import configure_json_logging` — imports the shared handler-setup function documented in `logging_setup.py`. This file doesn't implement its own logging configuration; it delegates to that one shared implementation.

### Lines 11-16 — Logger name and module-level logger object
```python
# Matches evaluation/runner.py's module path, the same convention
# sync_log.py follows for ingestion/scheduler.py.
LOGGER_NAME = "agentic_rag.evaluation.runner"

_logger = logging.getLogger(LOGGER_NAME)
```
- The comment explains a naming convention used across the `observability/*_log.py` family: each module's logger is named after the *other* module (the one that actually triggers the logging) that it is instrumenting — here, `evaluation/runner.py`. This means if that module also does its own plain `logger.info(...)` calls via `logging.getLogger(__name__)`, they'd share the exact same configured logger and output stream.
- `LOGGER_NAME = "agentic_rag.evaluation.runner"` — defines that name as a constant so it's used consistently in both the configuration function and the module-level logger below, rather than being retyped (and risking a typo-driven mismatch) in multiple places.
- `_logger = logging.getLogger(LOGGER_NAME)` — fetches (or creates) the logger object with that name once, at import time, and stores it in a module-private variable (`_logger`, leading underscore signaling "internal use only") so the logging functions below don't need to look it up again on every call.

### Lines 18-26 — `configure_eval_logging()`
```python
def configure_eval_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.evaluation.runner` logger at `stream`
    (default: the current `sys.stdout`) as one structured JSON line per
    `python -m agentic_rag.evaluation.runner` run. Thin wrapper around
    `logging_setup.configure_json_logging()` - see that function's
    docstring for the idempotency/stdout-timing reasoning shared by
    every `observability/*_log.py` module.
    """
    configure_json_logging(LOGGER_NAME, stream=stream)
```
- `def configure_eval_logging(*, stream: TextIO | None = None) -> None:` — the public setup function callers (like the evaluation runner's `main()`) invoke once before logging anything, to wire up where output goes. `stream` is keyword-only (the `*`) and optional, defaulting to `None` so the "current `sys.stdout`" resolution happens inside `configure_json_logging`, not here.
- The docstring is short because it defers to `logging_setup.py`'s own docstring for the detailed reasoning (idempotency — safe to call more than once — and the "resolve stdout now, not at import time" behavior) rather than repeating it.
- `configure_json_logging(LOGGER_NAME, stream=stream)` — the entire function body: it just forwards to the shared helper with this module's specific logger name. This is the "thin wrapper" pattern used consistently across all four `*_log.py` modules.

### Lines 29-38 — `log_evaluation_run()` signature
```python
def log_evaluation_run(
    *,
    retrieval_precision: float | None,
    faithfulness_rate: float | None,
    hallucination_rate: float | None,
    errored_count: int,
    average_duration_seconds: float | None,
    report_path: str,
    run_duration_seconds: float,
) -> None:
```
- All parameters are keyword-only (`*`), which forces every call site to name each argument explicitly (e.g. `retrieval_precision=0.8`) rather than relying on positional order — this makes call sites self-documenting and prevents an easy mistake like accidentally swapping two `float` arguments of the same type.
- `retrieval_precision`, `faithfulness_rate`, `hallucination_rate`, `average_duration_seconds` are typed `float | None` (a number or nothing) rather than plain `float` — explained in the docstring below.
- `errored_count: int` — how many evaluation questions in the run raised an error rather than being scored.
- `report_path: str` — the filesystem path where the full, detailed per-question report was already written by the caller.
- `run_duration_seconds: float` — how long the entire evaluation run took, wall-clock time.

### Lines 39-45 — Docstring: this log line vs. the full report file
```python
    """Emit one structured JSON log line summarizing a whole evaluation
    run (`evaluation/runner.py::main()`), alongside (not instead of) the
    full per-question JSON report `main()` already writes to
    `report_path` - this line is the live, glanceable summary signal in
    the same JSON-lines stream every other component logs through;
    the report file remains the detailed, per-question artifact for
    actually auditing a run's individual answers.
```
- Explains the relationship between this one log line and the separate, more detailed report file the evaluation runner writes to disk: the log line is a quick, aggregate summary meant for real-time monitoring (e.g. watching the log stream, or feeding it into a dashboard), while the full report file is where someone would go to inspect exactly which individual questions passed or failed and why. The two aren't duplicates of each other — they serve different purposes.

### Lines 47-53 — Docstring: why `None` (not `0.0`) for missing metrics
```python
    `retrieval_precision`/`faithfulness_rate`/`hallucination_rate`/
    `average_duration_seconds` are `float | None`, matching
    `EvaluationReport`'s own fields exactly (`evaluation/report.py`) -
    `None` means "no question in this run had that metric apply at all,"
    a fact distinct from "the metric was computed and came out 0," and
    coercing it to `0.0` here would destroy that distinction the report
    itself was deliberately built to preserve.
```
- This is a key design-decision explanation: a metric being `0.0` (e.g. zero hallucinations detected) is meaningfully different from a metric being inapplicable entirely (e.g. no question in the run was of a type that metric even measures). If this function silently converted a missing value to `0.0` before logging it, that distinction would be lost, and a reader of the logs might wrongly conclude "hallucination rate was perfect" when really "hallucination rate was never measured." The function preserves the `None` as-is so JSON output faithfully shows `null` in that case.

### Lines 55-59 — Docstring: `run_duration_seconds` vs. `average_duration_seconds`
```python
    `run_duration_seconds` is the whole run's wall-clock time (indexing
    the corpus plus every question), a coarser, separate number from
    `average_duration_seconds` (the mean of individual questions'
    `duration_seconds`, itself already inclusive of errored questions -
    see `report.py::build_report()`'s own docstring).
```
- Clarifies that these two duration-related fields measure different things and shouldn't be confused: `run_duration_seconds` covers the entire process end-to-end (including setup work like indexing), while `average_duration_seconds` is a per-question average that already accounts for questions that errored out.

### Lines 61-68 — Docstring: `report_id` derived from `report_path`
```python
    `report_id` (`Path(report_path).stem`, e.g. `eval-20260815T153329`)
    is included alongside `report_path` - `report_path` is a full,
    environment-specific filesystem path (different on every machine/
    container that runs an eval), while `report_id` is a portable
    correlation key that stays meaningful even if logs get shipped
    somewhere the local report file isn't reachable from. Derived from
    `report_path` rather than passed separately, so the two can never
    drift out of sync with each other.
    """
```
- Explains why the payload includes both a full path and a short identifier extracted from it: the full path is only meaningful on the machine that produced it, whereas the short `report_id` (the filename without its extension, e.g. `eval-20260815T153329`) remains a useful cross-reference even if the log line is shipped somewhere else (like a centralized logging service) where the original report file can't be opened directly. Deriving `report_id` from `report_path` inside this function (rather than requiring the caller to pass both separately) guarantees they can never accidentally disagree with each other.

### Lines 70-81 — Building the payload dictionary
```python
    payload = {
        "event": "evaluation_run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_precision": retrieval_precision,
        "faithfulness_rate": faithfulness_rate,
        "hallucination_rate": hallucination_rate,
        "errored_count": errored_count,
        "average_duration_seconds": average_duration_seconds,
        "report_id": Path(report_path).stem,
        "report_path": report_path,
        "run_duration_seconds": run_duration_seconds,
    }
```
- `payload = {...}` — assembles a plain Python dictionary holding every field that will end up in the JSON log line.
- `"event": "evaluation_run"` — a fixed, constant tag identifying what kind of event this log line represents, so downstream tooling/readers can filter for exactly this event type among a mixed stream of different log events.
- `"timestamp": datetime.now(timezone.utc).isoformat()` — records the current time in UTC (Coordinated Universal Time, a timezone-independent reference point), formatted as an ISO-8601 string, so log lines are comparable and sortable across machines/timezones without ambiguity.
- The remaining keys (`retrieval_precision` through `run_duration_seconds`) simply copy the function's parameters into the dictionary as-is — no transformation is applied to most of them, preserving the semantics (including `None` values) discussed above.
- `"report_id": Path(report_path).stem` — this is where the derived identifier from the docstring is actually computed: `Path(...).stem` takes a file path and returns just its filename without the directory or file extension.

### Lines 82-83 — Emitting the log line
```python
    _logger.info(json.dumps(payload))
```
- `json.dumps(payload)` — serializes the dictionary into a single-line JSON string.
- `_logger.info(...)` — sends that string through the logger configured earlier by `configure_eval_logging()`. Because that configuration set the formatter to emit `%(message)s` only (see `logging_setup.py`), this results in exactly one clean line of JSON written to the output stream, with nothing else appended.
