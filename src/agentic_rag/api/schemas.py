from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: Literal["ok"] = Field(
        description=(
            "Always \"ok\" when this endpoint responds at all - a non-200 status "
            "or a failed connection is the actual signal of an unhealthy process, "
            "not a field value to inspect."
        )
    )


class ReadinessResponse(BaseModel):
    """Response body for `GET /health/ready`.

    Distinct from `GET /health`: liveness ("is the process up") and
    readiness ("can this process actually serve a request right now")
    are different questions - a process can be alive but unable to serve
    traffic if Qdrant or Ollama is unreachable. `status`/HTTP status code
    both reflect the same fact so either a status-code-only check (a
    container orchestrator) or a body-reading check (a human, a richer
    monitoring tool) gets the right answer.
    """

    status: Literal["ready", "not_ready"] = Field(
        description="\"ready\" only if every dependency in `checks` succeeded."
    )
    checks: dict[str, str] = Field(
        description=(
            "One entry per dependency checked (currently \"qdrant\" and "
            "\"ollama\"). Each value is \"ok\", or the error that made the "
            "check fail - never omitted, so a caller always knows which "
            "specific dependency is the problem, not just that something is."
        )
    )


class ConversationTurnModel(BaseModel):
    """One prior turn of the conversation, as the client remembers it.

    `POST /query` is stateless (docs/REQUIREMENTS.md §13) - the server
    holds no session state, so the client resends the full history it
    wants considered on every call.
    """

    user_query: str = Field(description="The user's message for this prior turn.")
    assistant_answer: str = Field(
        description="The assistant's response for this prior turn, exactly as returned."
    )


class QueryRequest(BaseModel):
    """Request body for `POST /query`."""

    query: str = Field(min_length=1, description="The current turn's question.")
    user_tier: str = Field(
        min_length=1,
        description=(
            "The requesting user's access tier. Must be one of the tiers "
            "configured server-side; an unrecognized tier returns 422."
        ),
    )
    history: list[ConversationTurnModel] = Field(
        default=[],
        description="Prior turns of this conversation, oldest first. Empty for a new conversation.",
    )

    @field_validator("query")
    @classmethod
    def _reject_whitespace_only_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class CitationModel(BaseModel):
    """One resolved source for a `[N]` marker in `QueryResponse.answer`."""

    number: int = Field(
        description="The `[N]` marker this citation resolves, matching its position in `answer`."
    )
    relative_path: str = Field(
        description="Path of the source document, relative to the indexed corpus root."
    )
    chunk_index: int = Field(
        description="Index of the specific chunk within the source document that was cited."
    )
    access_tier: str = Field(description="The access tier the cited document belongs to.")


class QueryResponse(BaseModel):
    """Response body for `POST /query`."""

    answer: str = Field(
        description=(
            "The generated answer, with `[N]` markers for each cited source. May be "
            "the canonical fallback message if no grounded answer was found, or if "
            "either the query or the generated answer failed a security check."
        )
    )
    citations: list[CitationModel] = Field(
        description=(
            "Resolves every `[N]` marker in `answer` to its source. Empty when "
            "`answer` is the canonical fallback message."
        )
    )
