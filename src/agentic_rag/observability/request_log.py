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
VERDICT_ERROR = "error"

# Caps how much of `query`/`rewritten_query` a single log line can carry -
# QueryRequest.query has no max_length and no request-body-size limit
# exists anywhere in the app, so without this an oversized (or adversarial)
# query would produce a proportionally huge log line on every request, not
# just error paths.
_MAX_LOGGED_TEXT_LENGTH = 2000

_logger = logging.getLogger(LOGGER_NAME)
_active_handler: logging.Handler | None = None


def configure_request_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.query` logger at `stream` (default: the
    *current* `sys.stdout`, looked up when this function runs - not
    whatever `sys.stdout` happened to be when this module was first
    imported) as one structured JSON line per `POST /query` request.

    Resolving `sys.stdout` inside the function body, rather than as a
    `stream: TextIO = sys.stdout` parameter default, matters because a
    parameter default is evaluated exactly once, at import time - if
    anything reassigns `sys.stdout` afterwards (a test framework's output
    capture, a console encoding reconfiguration, a supervisor redirecting
    stdout post-import), a default bound at import time keeps writing to
    the stale original object forever, silently, with nothing to indicate
    the real current stdout stopped receiving these lines.

    Neither the root logger nor uvicorn's own logging config attaches a
    handler to application-level loggers by default - without this, a
    plain `logger.info(...)` call would go nowhere (Python's `lastResort`
    handler only surfaces `WARNING`+ to stderr). `propagate = False` stops
    the same line from also being formatted/emitted a second time by
    whatever the root logger's config happens to be.

    Idempotent: the one handler this function previously attached (tracked
    by direct reference, not by scanning for a marker attribute) is
    removed before a new one is added, so calling it more than once (app
    startup running twice in a test process, or re-pointing at a different
    stream) never leaves the logger writing every line more than once.
    """
    global _active_handler

    if _active_handler is not None:
        _logger.removeHandler(_active_handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    _logger.addHandler(handler)
    _active_handler = handler
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def _truncate(text: str | None) -> str | None:
    if text is None or len(text) <= _MAX_LOGGED_TEXT_LENGTH:
        return text
    return text[:_MAX_LOGGED_TEXT_LENGTH] + f"...[truncated, {len(text)} chars total]"


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
    correctly declined as unanswerable, refused by output security, or
    `VERDICT_ERROR` for an unhandled exception the caller re-raises after
    logging) - a fixed vocabulary rather than free text, so downstream log
    queries ("how often do we refuse for injection vs. foul language?")
    don't have to fuzzy-match message strings.

    `timings_seconds` is an open dict rather than fixed fields, since
    which phases actually ran (and are worth timing) differs by
    `verdict` - an early refusal only has a `screen_input` phase plus
    `total`, while a fully answered request also has `rewrite`, `answer`,
    and `output_security`.

    `query`/`rewritten_query` are truncated (see `_MAX_LOGGED_TEXT_LENGTH`)
    before being written - see that constant's comment for why.
    """
    payload = {
        "event": "query_request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_tier": user_tier,
        "query": _truncate(query),
        "rewritten_query": _truncate(rewritten_query),
        "history_turns": history_turns,
        "verdict": verdict,
        "retrieval_hit_count": retrieval_hit_count,
        "cited_paths": cited_paths,
        "timings_seconds": timings_seconds,
    }
    _logger.info(json.dumps(payload))
