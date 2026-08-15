"""ASGI entry point for running the app as a standalone server.

`create_app()` (`agentic_rag/api/app.py`) deliberately takes `Settings`
explicitly rather than constructing one internally, so tests can point it
at an ephemeral `tmp_path` corpus/Qdrant instance instead of whatever
`.env`/environment happens to be present. A real deployment has no such
test-specific `Settings` to pass in - this module is the one place that
gap gets closed: `uvicorn agentic_rag.api.main:create --factory` (or
Docker's `CMD`) imports this module and calls `create()`, which builds
`Settings()` from the environment/.env the same way every other entry
point in this codebase already does (`evaluation/runner.py::main()`).

A factory function, not a bare module-level `app = create_app(Settings())`
- the latter would run `Settings()` validation as a side effect of merely
*importing* this module, so any incidental import (a docs generator,
static analysis, a future blanket-import test) would crash with a
`pydantic.ValidationError` even though nothing about actually running the
app was intended. `create()` only runs when something explicitly calls
it (uvicorn's `--factory` flag does exactly that, once, at startup),
while still failing fast the moment it *is* called: a required field
missing (`watched_folder_path` has no default) raises immediately, on
startup, instead of surfacing confusingly on the first request.
"""

from fastapi import FastAPI

from agentic_rag.api.app import create_app
from agentic_rag.config import Settings


def create() -> FastAPI:
    """Factory for `uvicorn agentic_rag.api.main:create --factory`."""
    return create_app(Settings())
