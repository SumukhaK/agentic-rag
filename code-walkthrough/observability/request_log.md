# `observability/request_log.py`

**Purpose:** This file is responsible for producing exactly one structured (JSON, one self-contained event per line) log line for every `POST /query` request the API handles — the endpoint where a user asks the football intelligence assistant a question. Because a single request can end in several different ways (answered normally, refused for containing a prompt-injection attempt, refused for foul language, correctly declined as unanswerable from the available data, refused by an output-safety check, or failed with an unhandled error), this module defines a fixed vocabulary of outcomes ("verdicts") and a fixed set of fields so that every one of those outcomes is logged consistently and can be queried/aggregated later (e.g. "how often do we refuse for injection versus foul language?"). Like the other `observability/*_log.py` modules, it builds on the shared logger-configuration helper in `logging_setup.py` rather than reimplementing that setup logic.

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
- `from __future__ import annotations` — enables the modern, lazily-evaluated type hint syntax used throughout this file (e.g. `str | None`), without runtime cost for hints nothing inspects.
- `import json` — used to convert the payload dictionary built by `log_query_request()` into a JSON string before it's logged.
- `import logging` — Python's standard logging library, used to get the `Logger` object this module writes through.
- `from datetime import datetime, timezone` — used to stamp the log line with the current time in UTC.
- `from typing import TextIO` — the type used for the optional `stream` parameter passed into the configuration function.
- `from agentic_rag.observability.logging_setup import configure_json_logging` — imports the one shared helper that actually attaches and configures the logging handler; this file only supplies its own logger name and event shape.

### Lines 10-17 — Logger name and the fixed vocabulary of verdicts
```python
LOGGER_NAME = "agentic_rag.query"

VERDICT_ANSWERED = "answered"
VERDICT_REFUSED_INJECTION = "refused_injection"
VERDICT_REFUSED_FOUL_LANGUAGE = "refused_foul_language"
VERDICT_CANNOT_ANSWER = "cannot_answer"
VERDICT_REFUSED_OUTPUT_SECURITY = "refused_output_security"
VERDICT_ERROR = "error"
```
- `LOGGER_NAME = "agentic_rag.query"` — the name given to this module's logger, used both when configuring it and when fetching it below.
- The six `VERDICT_*` constants define every possible way a `POST /query` request can end, as plain string values: successfully `"answered"`, refused because the input looked like a prompt-injection attempt (`"refused_injection"`), refused for containing foul language (`"refused_foul_language"`), correctly declined because the assistant genuinely can't answer from the data it has (`"cannot_answer"`), refused because the generated output failed a safety check before being returned (`"refused_output_security"`), or `"error"` for an unhandled exception. Defining these as named constants (rather than callers typing the raw strings themselves) means a typo like `"refused_injecton"` becomes an `AttributeError`/`NameError` caught immediately, instead of a silent mismatch that would fragment the same logical outcome into two different strings in the logs.

### Lines 19-26 — Truncation limit and its rationale
```python
# Caps how much of `query`/`rewritten_query` a single log line can carry -
# QueryRequest.query has no max_length and no request-body-size limit
# exists anywhere in the app, so without this an oversized (or adversarial)
# query would produce a proportionally huge log line on every request, not
# just error paths.
_MAX_LOGGED_TEXT_LENGTH = 2000

_logger = logging.getLogger(LOGGER_NAME)
```
- The comment explains a defensive design choice: because the API's request model (`QueryRequest.query`) doesn't enforce any maximum length, and there's no overall limit on how large an incoming request body can be, a user (accidentally or maliciously) sending an enormous amount of text as their "question" would otherwise cause every single logged request — not just error cases — to produce an equally enormous log line. `_MAX_LOGGED_TEXT_LENGTH = 2000` sets a hard cap (2000 characters) on how much of the query text actually gets written into the log, protecting log storage and readability regardless of what a client sends.
- `_logger = logging.getLogger(LOGGER_NAME)` — fetches (or creates) the logger object for `"agentic_rag.query"` once, storing it for reuse by the functions below.

### Lines 29-36 — `configure_request_logging()`
```python
def configure_request_logging(*, stream: TextIO | None = None) -> None:
    """Point the `agentic_rag.query` logger at `stream` (default: the
    current `sys.stdout`) as one structured JSON line per `POST /query`
    request. Thin wrapper around `logging_setup.configure_json_logging()`
    - see that function's docstring for the idempotency/stdout-timing
    reasoning shared by every `observability/*_log.py` module.
    """
    configure_json_logging(LOGGER_NAME, stream=stream)
```
- `def configure_request_logging(*, stream: TextIO | None = None) -> None:` — the setup function run once (typically at application startup) to wire this logger to an output stream. `stream` is keyword-only and defaults to `None` so `configure_json_logging` resolves "the current `sys.stdout`" itself, at call time.
- The docstring defers to `logging_setup.py` for the detailed idempotency (safe to call repeatedly without duplicating output) and stdout-timing reasoning, since it's identical to the other `*_log.py` modules.
- `configure_json_logging(LOGGER_NAME, stream=stream)` — the entire body: delegate to the shared setup helper with this module's specific logger name.

### Lines 39-42 — `_truncate()` helper
```python
def _truncate(text: str | None) -> str | None:
    if text is None or len(text) <= _MAX_LOGGED_TEXT_LENGTH:
        return text
    return text[:_MAX_LOGGED_TEXT_LENGTH] + f"...[truncated, {len(text)} chars total]"
```
- `def _truncate(text: str | None) -> str | None:` — a small private helper function (leading underscore signals it's internal to this module) that enforces the length cap defined above. It accepts either a string or `None` and returns the same type.
- `if text is None or len(text) <= _MAX_LOGGED_TEXT_LENGTH: return text` — if there's no text to truncate, or it's already within the limit, it's returned completely unchanged.
- `return text[:_MAX_LOGGED_TEXT_LENGTH] + f"...[truncated, {len(text)} chars total]"` — otherwise, the text is cut down to the first 2000 characters, and a marker is appended showing it was truncated along with the *original* full length — so a reader of the log can tell both that truncation happened and how much text they're not seeing, rather than a truncated string silently looking complete.

### Lines 45-56 — `log_query_request()` signature
```python
def log_query_request(
    *,
    request_id: str,
    user_tier: str,
    query: str,
    rewritten_query: str | None,
    history_turns: int,
    verdict: str,
    retrieval_hit_count: int,
    cited_paths: list[str],
    timings_seconds: dict[str, float],
) -> None:
```
- All parameters are keyword-only, forcing call sites to name each one explicitly.
- `request_id: str` — a unique identifier for this one request (a UUID, generated fresh in `api/routers/query.py` before anything else runs and passed straight through). See the dedicated docstring section below for why this exists.
- `user_tier: str` — which tier/plan the requesting user belongs to (used elsewhere in the app to control things like rate limits); recorded here so usage patterns can be broken down by tier.
- `query: str` — the user's original question text, as submitted.
- `rewritten_query: str | None` — the query after being rewritten/reformulated (e.g. to incorporate conversation history), or `None` if that step never ran (explained below).
- `history_turns: int` — how many prior turns of conversation history were considered for this request.
- `verdict: str` — one of the `VERDICT_*` constants, describing how the request ended.
- `retrieval_hit_count: int` — how many documents/chunks were retrieved from the knowledge base for this query.
- `cited_paths: list[str]` — the specific source document paths the final answer actually cited.
- `timings_seconds: dict[str, float]` — an open-ended mapping of phase name to how long that phase took (see below for why this is a dict rather than fixed fields).

### Lines 56-64 — Docstring: one line per request, built after the outcome is known
```python
    """Emit one structured JSON log line summarizing a `POST /query`
    request end to end.

    One call per request, made once the outcome is known - not scattered
    `logger.info()` calls at each step - so a single log line always
    tells the whole story of one request rather than requiring a reader
    to reconstruct it from several interleaved lines under concurrent
    load.
```
- Explains a deliberate structural choice: rather than logging progress incrementally as a request moves through each processing step (which, under concurrent load with many requests being handled in parallel, would produce many interleaved lines from different requests that a reader would have to painstakingly untangle), this function is called exactly once, after the entire request's outcome is already known. That guarantees each log line is self-contained and tells the complete story of one request by itself.

### Lines 66-73 — Docstring: why `request_id` exists
```python
    `request_id` (a UUID minted once per request in `api/routers/query.py`,
    not derived from anything in `payload`) exists specifically for that
    "concurrent load" case: two requests with identical `query`/`user_tier`
    arriving close together produce log lines a reader can't otherwise tell
    apart, and nothing in this line's other fields lets a reader correlate
    it with, say, a downstream error logged elsewhere for the same request.
    Minted server-side, not accepted from the client, so it can't be
    spoofed or reused across requests by a caller.
```
- Explains what problem `request_id` actually solves: the paragraph just above already established that logging once per request (not incrementally) avoids interleaved partial lines, but that alone still doesn't solve *this* problem — two genuinely different requests can have identical `user_tier`/`query` values (the same person asking the same question twice, or two different people asking the same thing at once), so without something unique per request, a reader still can't tell two such log lines apart or know for certain which is which if they need to cross-reference against, say, an error logged somewhere else for one specific request.
- The ID is generated with Python's `uuid` module on the server side, inside `api/routers/query.py`, before any other work happens for that request — it's never something a client sends in and the server merely echoes back. That matters for trustworthiness: a client-supplied ID could be reused across multiple different requests (defeating the whole point) or deliberately collided with another request's ID.

### Lines 74-78 — Docstring: why `rewritten_query` can be `None`
```python
    `rewritten_query` is `None`, not the raw `query` duplicated, for a
    request refused by input screening (`_screen_input()` in
    `api/routers/query.py`) before `rewrite_query()` ever ran - collapsing
    that into a placeholder value would hide that the rewrite step was
    skipped entirely, not merely a no-op.
```
- Explains why `rewritten_query` is nullable rather than always holding some string value: if a request gets refused by an early input-screening check (in `api/routers/query.py`) before the query-rewriting step even runs, logging some placeholder value (like duplicating the original `query`) would misleadingly suggest the rewrite step ran and simply didn't change anything, when in fact it never executed at all. Keeping it as `None` in that case preserves an accurate record of what actually happened.

### Lines 71-77 — Docstring: `verdict` as a fixed vocabulary
```python
    `verdict` is one of the `VERDICT_*` constants above, covering every
    way a request can end (answered, refused by either input screen,
    correctly declined as unanswerable, refused by output security, or
    `VERDICT_ERROR` for an unhandled exception the caller re-raises after
    logging) - a fixed vocabulary rather than free text, so downstream log
    queries ("how often do we refuse for injection vs. foul language?")
    don't have to fuzzy-match message strings.
```
- Reiterates why `verdict` is restricted to the six defined constants rather than arbitrary free-text strings: it makes later analysis of the logs reliable — someone can search for exactly `"refused_injection"` and know they've found every matching event, instead of needing to guess at and fuzzy-match however many different phrasings a free-text message might have used over time. It also notes that `VERDICT_ERROR` is used when an unhandled exception occurs, and that in that case the caller is expected to log this event and then re-raise the exception (not swallow it).

### Lines 79-83 — Docstring: `timings_seconds` as an open dict
```python
    `timings_seconds` is an open dict rather than fixed fields, since
    which phases actually ran (and are worth timing) differs by
    `verdict` - an early refusal only has a `screen_input` phase plus
    `total`, while a fully answered request also has `rewrite`, `answer`,
    and `output_security`.
```
- Explains why `timings_seconds` is a flexible dictionary (mapping arbitrary phase names to durations) rather than a fixed set of named fields like `screen_input_seconds`, `rewrite_seconds`, etc.: different verdicts pass through genuinely different sets of processing phases. A request refused immediately at input screening only ever has a `screen_input` phase and an overall `total`, whereas a request that goes all the way through to a successful answer also has `rewrite`, `answer`, and `output_security` phases timed. A fixed-fields schema would force every log line to carry irrelevant zero/null fields for phases that never ran; the open dict avoids that.

### Lines 85-87 — Docstring: truncation applies here too
```python
    `query`/`rewritten_query` are truncated (see `_MAX_LOGGED_TEXT_LENGTH`)
    before being written - see that constant's comment for why.
    """
```
- A final note tying the function back to the `_truncate()` helper and the `_MAX_LOGGED_TEXT_LENGTH` constant defined earlier, confirming both text fields go through truncation before being logged.

### Lines 88-100 — Building the payload dictionary
```python
    payload = {
        "event": "query_request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "user_tier": user_tier,
        "query": _truncate(query),
        "rewritten_query": _truncate(rewritten_query),
        "history_turns": history_turns,
        "verdict": verdict,
        "retrieval_hit_count": retrieval_hit_count,
        "cited_paths": cited_paths,
        "timings_seconds": timings_seconds,
    }
```
- `payload = {...}` — assembles the dictionary that will become the JSON log line.
- `"event": "query_request"` — a fixed tag identifying this specific kind of log event, distinguishing it from other event types (like `evaluation_run` or `sync_cycle`) that might appear in a combined log stream.
- `"timestamp": datetime.now(timezone.utc).isoformat()` — the current time in UTC, in ISO-8601 string form, matching the convention used across all the other `*_log.py` modules for consistent, timezone-unambiguous sorting/comparison.
- `"request_id": request_id` — copied straight through from the parameter with no transformation; see the dedicated docstring section above for what this is for.
- `"query": _truncate(query)` and `"rewritten_query": _truncate(rewritten_query)` — both text fields are passed through the truncation helper before being stored, applying the length cap discussed earlier.
- The rest of the keys (`user_tier`, `history_turns`, `verdict`, `retrieval_hit_count`, `cited_paths`, `timings_seconds`) are copied directly from the function's parameters without transformation.

### Lines 100 — Emitting the log line
```python
    _logger.info(json.dumps(payload))
```
- `json.dumps(payload)` — serializes the payload dictionary into a single JSON-formatted string.
- `_logger.info(...)` — writes that string through the logger configured by `configure_request_logging()`. Because that setup used the `%(message)s`-only formatter from `logging_setup.py`, the result is exactly one clean line of JSON per request, with no other text mixed in.
