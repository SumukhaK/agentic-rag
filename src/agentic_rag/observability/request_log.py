from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import TextIO

LOGGER_NAME = "agentic_rag.query"

VERDICT_ANSWERED = "answered"
VERDICT_REFUSED_INJECTION = "refused_injection"
VERDICT_REFUSED_FOUL_LANGUAGE = "refused_foul_language"
VERDICT_CANNOT_ANSWER = "cannot_answer"
VERDICT_REFUSED_OUTPUT_SECURITY = "refused_output_security"

_HANDLER_MARKER = "_agentic_rag_request_log_handler"

_logger = logging.getLogger(LOGGER_NAME)


def configure_request_logging(*, stream: TextIO = sys.stdout) -> None:
    """Point the `agentic_rag.query` logger at `stream` as one structured
    JSON line per `POST /query` request.

    Neither the root logger nor uvicorn's own logging config attaches a
    handler to application-level loggers by default - without this, a
    plain `logger.info(...)` call would go nowhere (Python's `lastResort`
    handler only surfaces `WARNING`+ to stderr). `propagate = False` stops
    the same line from also being formatted/emitted a second time by
    whatever the root logger's config happens to be.

    Idempotent: any handler this function previously attached is removed
    before a new one is added, so calling it more than once (app startup
    running twice in a test process, or re-pointing at a different
    stream) never leaves the logger writing every line more than once.
    """
    for handler in list(_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            _logger.removeHandler(handler)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _HANDLER_MARKER, True)

    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log_query_request(
    *,
    user_tier: str,
    query: str,
    rewritten_query: str | None,
    history_turns: int,
    verdict: str,
    retrieval_hit_count: int,
    cited_paths: list[str],
    timings_seconds: dict[str, float],
) -> None:
    """Emit one structured JSON log line summarizing a `POST /query`
    request end to end.

    One call per request, made once the outcome is known - not scattered
    `logger.info()` calls at each step - so a single log line always
    tells the whole story of one request rather than requiring a reader
    to reconstruct it from several interleaved lines under concurrent
    load.

    `rewritten_query` is `None`, not the raw `query` duplicated, for a
    request refused by input screening (`_screen_input()` in
    `api/routers/query.py`) before `rewrite_query()` ever ran - collapsing
    that into a placeholder value would hide that the rewrite step was
    skipped entirely, not merely a no-op.

    `verdict` is one of the `VERDICT_*` constants above, covering every
    way a request can end (answered, refused by either input screen,
    correctly declined as unanswerable, or refused by output security) -
    a fixed vocabulary rather than free text, so downstream log queries
    ("how often do we refuse for injection vs. foul language?") don't
    have to fuzzy-match message strings.

    `timings_seconds` is an open dict rather than fixed fields, since
    which phases actually ran (and are worth timing) differs by
    `verdict` - an early refusal only has a `screen_input` phase plus
    `total`, while a fully answered request also has `rewrite`, `answer`,
    and `output_security`.
    """
    payload = {
        "event": "query_request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_tier": user_tier,
        "query": query,
        "rewritten_query": rewritten_query,
        "history_turns": history_turns,
        "verdict": verdict,
        "retrieval_hit_count": retrieval_hit_count,
        "cited_paths": cited_paths,
        "timings_seconds": timings_seconds,
    }
    _logger.info(json.dumps(payload))
