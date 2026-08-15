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
    `POST /query` needs: Qdrant (via `collection_exists()`, the cheapest
    real call that proves the client can talk to the collection) and
    Ollama (via a lightweight request to its base URL, bounded by
    `settings.readiness_check_timeout_seconds` - deliberately short and
    separate from the generation/embedding timeouts, since this exists
    to answer "reachable right now," not to wait as long as a real
    embedding/generation call would).

    Every dependency is always checked, even after an earlier one fails
    - a caller debugging a `not_ready` response needs to see the full
    picture in one response, not just the first problem found.

    Returns 503 (not 200) when anything is unreachable, so a container
    orchestrator's readiness probe can act on the status code alone
    without parsing the body - the body's `checks` field is for a human
    or a richer monitoring tool to see *which* dependency is the
    problem.
    """
    checks: dict[str, str] = {}

    try:
        client.collection_exists(settings.qdrant_collection_name)
        checks["qdrant"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
        checks["qdrant"] = f"{type(exc).__name__}: {exc}"

    try:
        ollama_response = requests.get(
            settings.ollama_base_url, timeout=settings.readiness_check_timeout_seconds
        )
        ollama_response.raise_for_status()
        checks["ollama"] = "ok"
    except requests.RequestException as exc:
        checks["ollama"] = f"{type(exc).__name__}: {exc}"

    ready = all(value == "ok" for value in checks.values())
    response.status_code = 200 if ready else 503
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
