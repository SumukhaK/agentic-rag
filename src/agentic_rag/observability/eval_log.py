from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from agentic_rag.observability.logging_setup import configure_json_logging

# Matches evaluation/runner.py's module path, the same convention
# sync_log.py follows for ingestion/scheduler.py.
LOGGER_NAME = "agentic_rag.evaluation.runner"

_logger = logging.getLogger(LOGGER_NAME)


def configure_eval_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.evaluation.runner` logger at `stream`
    (default: the current `sys.stdout`) as one structured JSON line per
    `python -m agentic_rag.evaluation.runner` run. Thin wrapper around
    `logging_setup.configure_json_logging()` - see that function's
    docstring for the idempotency/stdout-timing reasoning shared by
    every `observability/*_log.py` module.
    """
    configure_json_logging(LOGGER_NAME, stream=stream)


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
    """Emit one structured JSON log line summarizing a whole evaluation
    run (`evaluation/runner.py::main()`), alongside (not instead of) the
    full per-question JSON report `main()` already writes to
    `report_path` - this line is the live, glanceable summary signal in
    the same JSON-lines stream every other component logs through;
    the report file remains the detailed, per-question artifact for
    actually auditing a run's individual answers.

    `retrieval_precision`/`faithfulness_rate`/`hallucination_rate`/
    `average_duration_seconds` are `float | None`, matching
    `EvaluationReport`'s own fields exactly (`evaluation/report.py`) -
    `None` means "no question in this run had that metric apply at all,"
    a fact distinct from "the metric was computed and came out 0," and
    coercing it to `0.0` here would destroy that distinction the report
    itself was deliberately built to preserve.

    `run_duration_seconds` is the whole run's wall-clock time (indexing
    the corpus plus every question), a coarser, separate number from
    `average_duration_seconds` (the mean of individual questions'
    `duration_seconds`, itself already inclusive of errored questions -
    see `report.py::build_report()`'s own docstring).

    `report_id` (`Path(report_path).stem`, e.g. `eval-20260815T153329`)
    is included alongside `report_path` - `report_path` is a full,
    environment-specific filesystem path (different on every machine/
    container that runs an eval), while `report_id` is a portable
    correlation key that stays meaningful even if logs get shipped
    somewhere the local report file isn't reachable from. Derived from
    `report_path` rather than passed separately, so the two can never
    drift out of sync with each other.
    """
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
    _logger.info(json.dumps(payload))
