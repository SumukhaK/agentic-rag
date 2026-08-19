# `api/main.py`

**Purpose:** This is the file a real server deployment (or `uvicorn`, the tool used to actually run a FastAPI app) points to when starting the application for real, as opposed to running it inside a test. Its entire job is to bridge the gap between `create_app()` in `api/app.py` (which deliberately expects a `Settings` object to be handed to it, so tests can supply a fake, temporary one) and a real deployment, where nobody is manually constructing that `Settings` object — it needs to be built from whatever's in the environment or `.env` file. This file provides a "factory" function that does exactly that construction, and does it lazily (only when actually called), so merely importing this module never has the side effect of validating configuration or crashing.

## Line-by-line walkthrough

### Lines 1-23 — Module docstring
```python
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
```
- This module-level docstring (a description string documenting the whole file, placed at the very top) explains the file's role as the "ASGI entry point" — the piece a web server (ASGI is the standard interface Python web servers like `uvicorn` use to talk to an application) loads to start the app for real.
- It explains why `create_app()` takes an explicit `Settings` argument instead of building its own: this lets tests hand it a `Settings` pointed at a temporary, throwaway test folder/database instead of whatever real `.env` file happens to be present, so tests don't accidentally touch real data.
- It explains that a real deployment (via `uvicorn agentic_rag.api.main:create --factory`, or a Docker container's startup command) has no test harness supplying a `Settings` object — so this module exists specifically to build one from the real environment, the same way the evaluation runner (`evaluation/runner.py`) does for its own entry point.
- It explains the key design decision: this is written as a *function* (`create()`) that gets called, rather than as a plain module-level line like `app = create_app(Settings())`. If it were a plain module-level line, simply *importing* this file — for any reason, even an unrelated tool like a documentation generator or a test that imports every module in the codebase — would immediately try to build a `Settings()` object and crash with a validation error if required configuration (like `watched_folder_path`, which has no default) were missing, even though nobody actually wanted to start the app. Wrapping it in a function means the validation, and the possible crash, only happens when something deliberately calls `create()` — which `uvicorn`'s `--factory` flag does exactly once, at real startup — and at that point, failing immediately (rather than only once the first request comes in) is exactly the desired behavior.

### Lines 25-28 — Imports
```python
from fastapi import FastAPI

from agentic_rag.api.app import create_app
from agentic_rag.config import Settings
```
- `from fastapi import FastAPI` — imports FastAPI's app class, used here only as the return-type annotation on `create()`.
- `from agentic_rag.api.app import create_app` — imports the actual app-building function defined in `api/app.py`, which this file will call.
- `from agentic_rag.config import Settings` — imports the `Settings` class so this file can construct one from the environment.

### Lines 31-33 — The `create` factory function
```python
def create() -> FastAPI:
    """Factory for `uvicorn agentic_rag.api.main:create --factory`."""
    return create_app(Settings())
```
- `def create() -> FastAPI:` — defines the factory function, named simply `create`, matching the command shown in the module docstring (`uvicorn agentic_rag.api.main:create --factory`) that a real deployment uses to launch the app.
- `"""Factory for `uvicorn agentic_rag.api.main:create --factory`."""` — a short docstring restating exactly how this function is meant to be invoked in production.
- `return create_app(Settings())` — the whole body: it constructs a fresh `Settings()` object (which reads from environment variables and `.env`, as configured in `config.py`) and immediately passes it into `create_app()` from `api/app.py`, returning the fully built FastAPI application. Because `Settings()` has no required arguments supplied here, if a required field like `watched_folder_path` is missing from the environment, this line is exactly where that failure would be raised — deliberately, so the problem surfaces at startup rather than later.
