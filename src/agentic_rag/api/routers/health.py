import requests
from fastapi import APIRouter, Depends, Response
from qdrant_client import QdrantClient

from agentic_rag.api.dependencies import get_qdrant_client, get_settings
from agentic_rag.api.schemas import HealthResponse, ReadinessResponse
from agentic_rag.config import Settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Liveness check - the process is up and serving requests."""
    return HealthResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    responses={503: {"model": ReadinessResponse, "description": "One or more dependencies unreachable"}},
)
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
    checks: dict[str, str] = {}
    failures: list[str] = []

    try:
        if not client.collection_exists(settings.qdrant_collection_name):
            raise RuntimeError(
                f"collection '{settings.qdrant_collection_name}' does not exist"
            )
        checks["qdrant"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
        checks["qdrant"] = f"{type(exc).__name__}: {exc}"
        failures.append("qdrant")

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

    ready = not failures
    response.status_code = 200 if ready else 503
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
