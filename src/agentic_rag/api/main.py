"""ASGI entry point for running the app as a standalone server.

`create_app()` (`agentic_rag/api/app.py`) deliberately takes `Settings`
explicitly rather than constructing one internally, so tests can point it
at an ephemeral `tmp_path` corpus/Qdrant instance instead of whatever
`.env`/environment happens to be present. A real deployment has no such
test-specific `Settings` to pass in - this module is the one place that
gap gets closed: `uvicorn agentic_rag.api.main:app` (or Docker's `CMD`)
imports this module, which builds `Settings()` from the environment/.env
the same way every other entry point in this codebase already does
(`evaluation/runner.py::main()`), and exposes the resulting app as a
module-level `app` object for uvicorn to serve.

Constructing `Settings()` at import time is deliberate: a required field
missing (`watched_folder_path` has no default) raises immediately, on
import/startup, instead of surfacing confusingly on the first request.
"""

from agentic_rag.api.app import create_app
from agentic_rag.config import Settings

app = create_app(Settings())
