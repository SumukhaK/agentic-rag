from fastapi import APIRouter

from agentic_rag.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Liveness check - the process is up and serving requests."""
    return HealthResponse(status="ok")
