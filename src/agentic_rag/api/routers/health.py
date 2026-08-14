from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check - the process is up and serving requests."""
    return {"status": "ok"}
