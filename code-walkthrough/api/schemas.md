# `api/schemas.py`

**Purpose:** This file defines the "shapes" of data that flow in and out of the API's HTTP endpoints, using Pydantic models (Python classes that describe a data structure's fields, their types, and validation rules, and that FastAPI uses to automatically validate incoming requests and document outgoing responses). Every request body and response body the API accepts or returns is defined here as a class, so the rest of the code — and anyone reading the auto-generated API documentation — has one authoritative, precise description of what a `/health`, `/health/ready`, or `/query` request or response actually looks like.

## Line-by-line walkthrough

### Lines 1-3 — Imports
```python
from typing import Literal

from pydantic import BaseModel, Field, field_validator
```
- `from typing import Literal` — imports `Literal`, a typing tool that restricts a field to one exact, specific set of allowed values (for example, only the string `"ok"`, and nothing else), rather than any string.
- `from pydantic import BaseModel, Field, field_validator` — imports `BaseModel` (the base class every schema below inherits from, giving it automatic validation and JSON conversion), `Field` (used to attach descriptions, defaults, and constraints to individual fields), and `field_validator` (a decorator used to write custom validation logic for a specific field).

### Lines 6-15 — `HealthResponse`
```python
class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: Literal["ok"] = Field(
        description=(
            "Always \"ok\" when this endpoint responds at all - a non-200 status "
            "or a failed connection is the actual signal of an unhealthy process, "
            "not a field value to inspect."
        )
    )
```
- `class HealthResponse(BaseModel):` — defines the response shape for the basic liveness check endpoint (`GET /health`), which just confirms the server process is running at all.
- `"""Response body for `GET /health`."""` — documents which endpoint this schema belongs to.
- `status: Literal["ok"] = Field(description=(...))` — declares a single field, `status`, whose type is restricted to exactly the string `"ok"` (via `Literal["ok"]`) — there is no other value it could ever hold. The attached description explains why: if this endpoint responds at all, it's always `"ok"` by definition; a caller who wants to know if the service is actually unhealthy should look at whether the HTTP call failed or returned a non-200 status, not inspect this field's value, since it can never be anything else.

### Lines 18-40 — `ReadinessResponse`
```python
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
```
- `class ReadinessResponse(BaseModel):` — defines the response shape for the readiness check endpoint (`GET /health/ready`), which answers a different question than `/health`: not just "is the process running" but "can it actually handle a real request right now."
- The class docstring explains why this is a separate concept from liveness: a process can be up and running but still unable to serve a request if one of its dependencies (Qdrant, the vector database, or Ollama, the local LLM server) is unreachable. It also explains that both the `status` field and the HTTP status code carry the same information redundantly, so that a simple automated check (like a container orchestrator that only looks at the HTTP status code) and a more detailed check (a human or monitoring tool reading the response body) both get a correct answer.
- `status: Literal["ready", "not_ready"] = Field(description=...)` — declares the overall verdict field, restricted to exactly two possible string values, with a description clarifying it's only `"ready"` if every single dependency check passed.
- `checks: dict[str, str] = Field(description=...)` — declares a dictionary field mapping each checked dependency's name (currently `"qdrant"` and `"ollama"`) to either `"ok"` or a description of the error that made it fail. The description explains this is deliberately always fully populated (never leaving a dependency out) so a caller can see exactly which dependency is the problem, not merely that something, somewhere, is wrong.

### Lines 43-49 — `ConversationTurnModel`
```python
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
```
- `class ConversationTurnModel(BaseModel):` — defines the shape of one past exchange in a conversation (one question and its answer), used as part of the conversation history a client can send along with a new query.
- The docstring explains an important architectural fact: the `/query` endpoint is stateless, meaning the server does not remember anything between requests on its own. Instead, if the client wants earlier turns of the conversation taken into account, it must resend that whole history with every single request.
- `user_query: str = Field(description="The user's message for this prior turn.")` — the text of what the user asked in that earlier turn.
- `assistant_answer: str = Field(description="...")` — the text the assistant answered with in that earlier turn, described as being exactly what was returned before (i.e., not paraphrased or altered by the client).

### Lines 57-78 — `QueryRequest`
```python
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
```
- `class QueryRequest(BaseModel):` — defines the shape of the body a client must send to `POST /query` to ask a question.
- `query: str = Field(min_length=1, description="The current turn's question.")` — the user's current question, required to be at least 1 character long (this alone doesn't stop pure-whitespace input, which is why the validator below exists).
- `user_tier: str = Field(min_length=1, description=(...))` — the access tier the requesting user belongs to; the description notes it must match one of the tiers configured on the server (see `access_tiers` in `config.py`), and that an unrecognized value results in an HTTP 422 (unprocessable request) error.
- `history: list[ConversationTurnModel] = Field(default=[], description=(...))` — a list of prior conversation turns, using the `ConversationTurnModel` shape defined above, defaulting to an empty list when the client is starting a brand-new conversation. The description notes the ordering convention: oldest turn first.
- `@field_validator("query")` — a decorator marking the function below it as a custom validation rule specifically for the `query` field, run automatically whenever a `QueryRequest` is constructed.
- `@classmethod` — marks the validator function as a class method (required by Pydantic's `field_validator` mechanism), meaning it receives the class itself (`cls`) rather than a particular instance.
- `def _reject_whitespace_only_query(cls, value: str) -> str:` — defines the validator function, named to describe exactly what it rejects.
- `if not value.strip(): raise ValueError("query must not be blank")` — checks whether the query, once leading/trailing whitespace is stripped away, is empty; if so, it raises a validation error, which Pydantic (and therefore FastAPI) turns into a 422 response. This closes the gap left by `min_length=1` alone, since a string made only of spaces would satisfy `min_length=1` but still be meaningless as a question.
- `return value` — if the check passes, the validator must return the (unmodified) value for Pydantic to actually use it.

### Lines 81-93 — `CitationModel`
```python
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
```
- `class CitationModel(BaseModel):` — defines the shape of a single citation: a record explaining exactly which source document (and which piece of it) backs up a `[N]`-style marker embedded in a generated answer.
- `number: int = Field(description=(...))` — the numeric marker (like the `1` in `[1]`) that this citation resolves, matching where it appears in the answer text.
- `relative_path: str = Field(description=(...))` — the path to the source document that was cited, given relative to the root of the indexed document corpus (not an absolute filesystem path, which would leak local machine details).
- `chunk_index: int = Field(description=(...))` — since documents are split into chunks before indexing (per `chunk_size_chars` in `config.py`), this identifies which specific chunk within that document was the actual source.
- `access_tier: str = Field(description="The access tier the cited document belongs to.")` — records which access tier the cited document is restricted to, letting a caller see the sensitivity level of the material an answer relied on.

### Lines 96-112 — `QueryResponse`
```python
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
```
- `class QueryResponse(BaseModel):` — defines the overall shape of what `POST /query` sends back to the caller.
- `answer: str = Field(description=(...))` — the generated answer text itself, containing `[N]`-style markers pointing at citations. Its description clarifies that this field can also hold a fixed fallback message instead of a real answer, in two situations: no answer could actually be grounded in the retrieved documents, or either the incoming query or the generated answer was flagged by a security check.
- `citations: list[CitationModel] = Field(description=(...))` — the list of `CitationModel` entries that resolve each `[N]` marker in `answer` to an actual source document and chunk. The description notes this list is empty whenever `answer` is the fallback message, since there's nothing real to cite in that case.
