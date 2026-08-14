from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: Literal["ok"] = Field(
        description=(
            "Always \"ok\" when this endpoint responds at all - a non-200 status "
            "or a failed connection is the actual signal of an unhealthy process, "
            "not a field value to inspect."
        )
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Liveness check - the process is up and serving requests."""
    return HealthResponse(status="ok")
