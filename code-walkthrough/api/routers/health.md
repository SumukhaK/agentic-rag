# `api/routers/health.py`

**Purpose:** This file defines the two health-check HTTP endpoints (`GET /health` and `GET /health/ready`) that let external systems — a container orchestrator, a load balancer, a human operator, or a monitoring tool — ask "is this service up?" and "is this service actually able to handle a request right now?" These are two genuinely different questions: a process can be alive (running, listening for requests) while still unable to do useful work because one of its dependencies, Qdrant (the vector database) or Ollama (the local LLM server), is unreachable. Splitting them into two endpoints lets automated systems, like Kubernetes, use the right one for the right purpose (for instance, restarting a process only for failed liveness, but temporarily routing traffic away for failed readiness).

## Line-by-line walkthrough

### Lines 1-9 — Imports and router setup
```python
import requests
from fastapi import APIRouter, Depends, Response
from qdrant_client import QdrantClient

from agentic_rag.api.dependencies import get_qdrant_client, get_settings
from agentic_rag.api.schemas import HealthResponse, ReadinessResponse
from agentic_rag.config import Settings

router = APIRouter()
```
- `import requests` — imports the `requests` library, used to make an HTTP call to Ollama's API to check if it's reachable.
- `from fastapi import APIRouter, Depends, Response` — imports `APIRouter` (a way to group related routes together, which the main app then mounts), `Depends` (FastAPI's dependency-injection marker, used to request a resource like settings or a database client without constructing it manually), and `Response` (lets a route handler directly control aspects of the raw HTTP response, like its status code).
- `from qdrant_client import QdrantClient` — imports the Qdrant client type, used as a type hint for the injected dependency.
- `from agentic_rag.api.dependencies import get_qdrant_client, get_settings` — imports the two dependency-provider functions (defined in `api/dependencies.py`) this file needs: one for the shared Qdrant client, one for the shared settings.
- `from agentic_rag.api.schemas import HealthResponse, ReadinessResponse` — imports the two Pydantic response models these endpoints will return, defined in `api/schemas.py`.
- `from agentic_rag.config import Settings` — imports the `Settings` type, used as a type hint for the injected settings dependency.
- `router = APIRouter()` — creates the router object that the two endpoints below will be registered on; `api/app.py` later mounts this router into the main FastAPI app.

### Lines 12-15 — The liveness endpoint
```python
@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Liveness check - the process is up and serving requests."""
    return HealthResponse(status="ok")
```
- `@router.get("/health", response_model=HealthResponse, summary="Liveness check")` — registers this function to handle `GET /health` requests, declares that its response will match the `HealthResponse` shape (so FastAPI can validate/document it), and gives it a human-readable summary for the auto-generated API docs.
- `def health() -> HealthResponse:` — the handler function itself; it takes no parameters because a liveness check needs no external information — the mere fact that this function can run and return at all is the answer.
- `"""Liveness check - the process is up and serving requests."""` — a docstring stating the endpoint's purpose plainly.
- `return HealthResponse(status="ok")` — always returns the fixed `"ok"` status; there's no other value it could return, since if the process couldn't respond, this code would never run at all.

### Lines 18-23 — The readiness endpoint's decorator
```python
@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    responses={503: {"model": ReadinessResponse, "description": "One or more dependencies unreachable"}},
)
```
- `@router.get("/health/ready", ...)` — registers the handler for `GET /health/ready`.
- `response_model=ReadinessResponse` — declares the normal (200) response shape.
- `summary="Readiness check"` — a short label for the API documentation.
- `responses={503: {"model": ReadinessResponse, "description": "..."}}` — explicitly documents that this endpoint can also return an HTTP 503 (Service Unavailable) status, using the same `ReadinessResponse` shape but meaning something different (a dependency failure) — so the auto-generated docs accurately show both possible outcomes, not just the successful one.

### Lines 24-63 — The readiness function's signature and docstring
```python
def readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
    client: QdrantClient = Depends(get_qdrant_client),
) -> ReadinessResponse:
    """Readiness check - can this process actually serve a request right
    now, not just "is it running." Checks the two dependencies every
    `POST /query` needs: Qdrant (via `collection_exists()`, checking the
    boolean result itself, not just whether the call raised - Qdrant
    being reachable but the configured collection missing is a real,
    distinct failure this must catch, not a call that "didn't error") and
    Ollama (via a request to `/api/tags`, the same real API surface a
    real Ollama deployment actually serves - not the bare base URL, which
    a reverse proxy or unrelated service could answer with a misleading
    200 with nothing behind it actually working). Bounded by
    `settings.readiness_check_timeout_seconds` - deliberately short and
    separate from the generation/embedding timeouts, since this exists
    to answer "reachable right now," not to wait as long as a real
    embedding/generation call would.

    Every dependency is always checked, even after an earlier one fails
    - a caller debugging a `not_ready` response needs to see the full
    picture in one response, not just the first problem found. Both
    checks catch the same broad `Exception` (not just
    `requests.RequestException` for Ollama) so an unusual failure mode
    degrades to a reported `checks` entry, the same as every other
    failure, rather than propagating into an unhandled 500 that defeats
    the "always see the full picture" guarantee this endpoint exists to
    provide.

    Returns 503 (not 200) when anything is unreachable, so a container
    orchestrator's readiness probe can act on the status code alone
    without parsing the body - the body's `checks` field is for a human
    or a richer monitoring tool to see *which* dependency is the
    problem. Readiness is tracked via `failures` (only ever appended to
    inside an `except` block), not by re-comparing `checks` values
    against the string `"ok"` - the latter would silently report "ready"
    for an empty `checks` dict, and depends on a literal string staying
    byte-identical in two separate places with nothing enforcing it.
    """
```
- `def readiness(response: Response, settings: Settings = Depends(get_settings), client: QdrantClient = Depends(get_qdrant_client)) -> ReadinessResponse:` — the function signature. `response: Response` lets the handler directly set the HTTP status code on the outgoing response (needed to return 503 rather than the default 200). `settings` and `client` are supplied automatically by FastAPI's dependency injection, using the getter functions from `api/dependencies.py`, rather than this function constructing or importing them directly — which keeps this route swappable/testable.
- The docstring explains readiness means "can this actually serve a request right now," distinct from mere liveness. It explains the two things checked: Qdrant, via `collection_exists()` — checking the actual boolean result, not just whether the call threw an exception, because Qdrant could be perfectly reachable while the specific collection this app needs is simply missing, which is its own distinct failure — and Ollama, via a real request to Ollama's `/api/tags` endpoint (an endpoint Ollama genuinely serves) rather than just the bare base URL, because something else entirely (like a reverse proxy) could answer a request to the bare URL with a misleadingly successful response even though Ollama itself isn't actually working behind it.
- It explains the timeout used (`settings.readiness_check_timeout_seconds`) is deliberately short and kept separate from the timeouts used for real embedding/generation calls, because this check only needs to answer "is it reachable right now," not wait as long as a genuine request would.
- It explains that both checks always run, even if the first one already failed, so that a caller looking at a `not_ready` response can see the complete picture (every problem) rather than just whichever check happened to run first.
- It explains why both checks catch the broad `Exception` type rather than a narrower one: an unusual error (of any kind) should still degrade gracefully into a reported failure in `checks`, rather than crash the whole endpoint with an unhandled server error, which would defeat the whole "see the full picture" purpose.
- It explains the endpoint returns HTTP 503, not 200, when anything is unreachable, so that automated tools that only look at the HTTP status code (not the JSON body) still get the right signal; the JSON body's `checks` field is there for humans or richer monitoring tools that want to know exactly which dependency is broken.
- It explains readiness is computed from a separate `failures` list (only ever added to inside error-handling code), rather than by scanning the `checks` dictionary afterward for the literal string `"ok"` — the latter approach would incorrectly report "ready" if `checks` happened to be empty, and would be fragile, since it would depend on the exact string `"ok"` staying identical everywhere it's used, with nothing to enforce that consistency.

### Lines 64-65 — Initializing the tracking structures
```python
    checks: dict[str, str] = {}
    failures: list[str] = []
```
- `checks: dict[str, str] = {}` — starts an empty dictionary that will record the result (`"ok"` or an error message) for each dependency checked.
- `failures: list[str] = []` — starts an empty list that will record the *names* of any dependencies that failed, used afterward purely to decide whether the overall status is "ready."

### Lines 67-75 — Checking Qdrant
```python
    try:
        if not client.collection_exists(settings.qdrant_collection_name):
            raise RuntimeError(
                f"collection '{settings.qdrant_collection_name}' does not exist"
            )
        checks["qdrant"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
        checks["qdrant"] = f"{type(exc).__name__}: {exc}"
        failures.append("qdrant")
```
- `try:` — begins a block that catches any error during the Qdrant check, so a Qdrant problem doesn't crash the whole readiness endpoint.
- `if not client.collection_exists(settings.qdrant_collection_name): raise RuntimeError(...)` — asks Qdrant whether the configured collection actually exists; if it doesn't, deliberately raises an error (rather than silently treating this as fine) so it's caught and reported the same way as any other failure.
- `checks["qdrant"] = "ok"` — if no exception occurred, records success.
- `except Exception as exc:  # noqa: BLE001 - report every failure, not just the first` — catches literally any exception type (the `# noqa: BLE001` comment tells the linter this deliberately broad exception catch is intentional, not an oversight, and explains why: to guarantee every failure gets reported rather than crashing the endpoint).
- `checks["qdrant"] = f"{type(exc).__name__}: {exc}"` — records the specific error type and message as the check's result, giving a caller concrete diagnostic detail rather than just "failed."
- `failures.append("qdrant")` — notes that this specific dependency failed, for the overall readiness calculation later.

### Lines 77-86 — Checking Ollama
```python
    try:
        ollama_response = requests.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=settings.readiness_check_timeout_seconds,
        )
        ollama_response.raise_for_status()
        checks["ollama"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
        checks["ollama"] = f"{type(exc).__name__}: {exc}"
        failures.append("ollama")
```
- `try:` — same pattern as above, isolating the Ollama check so a failure here doesn't take down the whole endpoint.
- `ollama_response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=settings.readiness_check_timeout_seconds)` — sends a real HTTP GET request to Ollama's `/api/tags` endpoint (which lists the models Ollama has available), bounded by the short readiness timeout, so this check can't hang the endpoint waiting on a slow or dead Ollama instance.
- `ollama_response.raise_for_status()` — raises an exception if Ollama responded with an HTTP error status (like 4xx or 5xx), so a technically-reachable-but-broken Ollama still counts as a failure, not a success.
- `checks["ollama"] = "ok"` — records success if the request succeeded and returned a good status.
- The `except` block mirrors the Qdrant one: catches any exception broadly, records the error type/message, and appends `"ollama"` to `failures`.

### Lines 88-90 — Computing and returning the overall result
```python
    ready = not failures
    response.status_code = 200 if ready else 503
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
```
- `ready = not failures` — the service is considered ready only if the `failures` list ended up empty (i.e., nothing failed).
- `response.status_code = 200 if ready else 503` — sets the actual HTTP status code on the response object directly: 200 if everything's fine, 503 (Service Unavailable) otherwise, matching the docstring's explanation of why this matters for automated tools.
- `return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)` — builds and returns the response body, with the `status` field reflecting the same "ready"/"not ready" determination as the HTTP status code, and `checks` carrying the full, per-dependency detail collected above.
