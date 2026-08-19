# `api/app.py`

**Purpose:** This file is where the FastAPI application itself gets assembled: it builds the `FastAPI` object, wires up startup/shutdown behavior (creating the shared Qdrant client and caches once, launching the background document-sync loop, and cleaning everything up on shutdown), mounts the health and query routers, and patches a gap in FastAPI's automatically generated API documentation. It's structured as a function, `create_app(settings)`, that takes configuration as an explicit argument rather than reading it globally — which is what lets tests build an app pointed at a temporary, isolated Qdrant database and corpus instead of whatever real data is configured in `.env`.

## Line-by-line walkthrough

### Lines 1-17 — Imports
```python
import asyncio
import importlib.metadata
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from agentic_rag.api.routers.health import router as health_router
from agentic_rag.api.routers.query import QUERY_422_DESCRIPTION
from agentic_rag.api.routers.query import router as query_router
from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.ingestion.scheduler import run_sync_loop
from agentic_rag.ingestion.snapshot_store import load_snapshot
from agentic_rag.observability.request_log import configure_request_logging
from agentic_rag.observability.sync_log import configure_sync_logging
from agentic_rag.orchestration.semantic_cache import SemanticCache
```
- `import asyncio` — imports Python's asynchronous programming library, used here to create and manage the background sync task as an `asyncio.Task`.
- `import importlib.metadata` — imports the standard library tool used to look up this package's own installed version number, so the API can report its real version rather than a hardcoded string.
- `from contextlib import asynccontextmanager, suppress` — imports `asynccontextmanager` (used to define the app's `lifespan` — code that runs once at startup and once at shutdown — as an async generator function) and `suppress` (a small helper for cleanly ignoring one specific, expected exception type without a full try/except block).
- `from fastapi import FastAPI` — imports the main FastAPI application class this file builds an instance of.
- `from agentic_rag.api.routers.health import router as health_router` — imports the health-check router (defined in `api/routers/health.py`), renamed locally to `health_router` for clarity.
- `from agentic_rag.api.routers.query import QUERY_422_DESCRIPTION` — imports a text constant from `api/routers/query.py` describing the `/query` endpoint's 422 error responses, used later to patch the OpenAPI schema.
- `from agentic_rag.api.routers.query import router as query_router` — imports the query router, renamed to `query_router`.
- `from agentic_rag.config import Settings` — imports the `Settings` type, used as the type hint for `create_app`'s parameter.
- `from agentic_rag.embedding.cache import EmbeddingCache` — imports the embedding cache class, one instance of which is created once per app and shared across all requests.
- `from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client` — imports `get_client` (which builds/connects the Qdrant client) and `ensure_collection` (which makes sure the expected collection exists with the right settings, failing fast if not).
- `from agentic_rag.ingestion.scheduler import run_sync_loop` — imports the function that runs the background loop responsible for periodically re-scanning the watched folder and updating the index.
- `from agentic_rag.ingestion.snapshot_store import load_snapshot` — imports the function used to load a previously saved sync snapshot from disk, so the sync loop doesn't have to start from scratch after a restart.
- `from agentic_rag.observability.request_log import configure_request_logging` — imports the function that sets up structured logging for query requests.
- `from agentic_rag.observability.sync_log import configure_sync_logging` — imports the equivalent setup function for logging sync-loop activity.
- `from agentic_rag.orchestration.semantic_cache import SemanticCache` — imports the semantic cache class, one instance of which is created once per app.

### Lines 20-70 — `create_app`'s signature and docstring
```python
def create_app(settings: Settings) -> FastAPI:
    """Build the Agentic RAG FastAPI app for `settings`.
    ...
    """
```
- `def create_app(settings: Settings) -> FastAPI:` — defines the app-building function, taking the already-constructed `Settings` object as its one required argument, and returning a ready-to-run `FastAPI` instance.
- The docstring explains several deliberate design choices in detail:
  - `settings` is passed in explicitly rather than the function building its own `Settings()` internally, matching a pattern used consistently throughout this codebase where every function takes its configuration as an argument instead of looking it up globally. This is also precisely what allows tests to build an app pointed at a temporary (`tmp_path`) Qdrant database and corpus, instead of whatever real `.env` file happens to exist on the machine running the tests.
  - The Qdrant client, embedding cache, and semantic cache are all created exactly once, inside the `lifespan` function below, and stored on `app.state` (a place to stash arbitrary shared objects on the app object) — not created fresh on every request. This matters because Qdrant's local/embedded mode locks its on-disk storage to a single client within a single process, so multiple clients can't coexist; and both caches are explicitly documented (in their own files) as only useful when they persist across the whole process's lifetime — a cache recreated per request would never have anything cached in it, defeating the entire purpose.
  - `ensure_collection` runs during startup specifically so that a mismatch between the app's expected schema and what's actually in Qdrant is caught immediately when the app boots, rather than confusingly surfacing later on the very first real query.
  - `lifespan` also starts the background sync loop (`run_sync_loop`, from `ingestion/scheduler.py`) as an `asyncio.Task` running inside this same process, rather than as a separate worker process — again because of the single-process, on-disk-locked Qdrant constraint, which rules out running sync somewhere else entirely. Its starting point is loaded from a persisted snapshot file (via `load_snapshot`) if one exists from a previous run, so a restart doesn't force a full, wasteful re-index of everything from scratch (`load_snapshot` simply returns an empty dictionary if no snapshot file exists yet).
  - It explains the shutdown sequence's structure: the sync task is cancelled and the Qdrant client is closed in nested `try`/`finally` blocks (not simply one after another as separate statements), which guarantees the Qdrant client's `close()` call runs no matter what happens while cancelling or awaiting the sync task — even if that awaiting itself raises something unexpected. It notes that `await sync_task` genuinely waits for the sync loop to actually stop mid-work, not merely for Python's coroutine wrapper around it to unwind, because the real sync work runs in a background thread (via `asyncio.to_thread()`), and cancelling a thread from the async side can't actually interrupt code already running inside that thread — so `run_sync_loop()` internally uses its own cooperative signaling (a `stop_event`/`done_event` pair) to actually stop cleanly rather than relying on cancellation alone.
  - It explains that `configure_request_logging()` and `configure_sync_logging()` are deliberately called here, inside `create_app()`, once per app — not automatically as a side effect of merely importing those modules — because a module-level side effect (like attaching a logging handler on import) would fire even for code that just wants to reuse logging helper functions or constants from those modules without ever wanting a handler auto-attached (for example, a standalone script, or a test). Because the sync loop runs in this same process, its logging also needs to be configured here, not only the query router's.

### Lines 71-72 — Configuring logging
```python
    configure_request_logging()
    configure_sync_logging()
```
- `configure_request_logging()` — sets up the logging configuration used when logging each `/query` request (see `observability/request_log.py`), run once per app as explained in the docstring above.
- `configure_sync_logging()` — sets up the equivalent logging configuration for the background sync loop (see `observability/sync_log.py`).

### Lines 74-97 — The `lifespan` context manager
```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = get_client(settings.qdrant_storage_path)
        ensure_collection(
            client, settings.qdrant_collection_name, settings.embedding_dimensions
        )
        app.state.settings = settings
        app.state.qdrant_client = client
        app.state.embedding_cache = EmbeddingCache()
        app.state.semantic_cache = SemanticCache()
        initial_snapshot = load_snapshot(settings.sync_snapshot_path)
        sync_task = asyncio.create_task(
            run_sync_loop(settings=settings, client=client, initial_snapshot=initial_snapshot)
        )
        app.state.sync_task = sync_task
        try:
            yield
        finally:
            try:
                sync_task.cancel()
                with suppress(asyncio.CancelledError):
                    await sync_task
            finally:
                client.close()
```
- `@asynccontextmanager` — a decorator that turns the async generator function below into something FastAPI can use as a "lifespan": code before the `yield` runs once at startup, and code after the `yield` (in the `finally` block) runs once at shutdown.
- `async def lifespan(app: FastAPI):` — defines the lifespan function, receiving the `FastAPI` app instance it's attached to.
- `client = get_client(settings.qdrant_storage_path)` — builds (or connects to) the Qdrant client pointed at the configured on-disk storage location.
- `ensure_collection(client, settings.qdrant_collection_name, settings.embedding_dimensions)` — verifies (and creates if necessary) that the expected Qdrant collection exists with the expected vector size, failing immediately if there's a mismatch, as explained in the docstring.
- `app.state.settings = settings` — stores the settings object on the app so route dependencies (`api/dependencies.py`) can retrieve it per-request via dependency injection.
- `app.state.qdrant_client = client` — stores the one shared Qdrant client on the app.
- `app.state.embedding_cache = EmbeddingCache()` — creates and stores the one shared embedding cache instance for this process's whole lifetime.
- `app.state.semantic_cache = SemanticCache()` — creates and stores the one shared semantic cache instance similarly.
- `initial_snapshot = load_snapshot(settings.sync_snapshot_path)` — loads whatever sync progress was saved from a previous run (or an empty result if there's no prior snapshot file).
- `sync_task = asyncio.create_task(run_sync_loop(settings=settings, client=client, initial_snapshot=initial_snapshot))` — starts the background document-sync loop as a concurrently running `asyncio` task, sharing this same process, passing in the settings, the Qdrant client, and the starting snapshot.
- `app.state.sync_task = sync_task` — stores a reference to the running task on the app state (so it can be found and cancelled during shutdown).
- `try: yield` — the point where control returns to FastAPI, which then serves requests for as long as the app runs; everything after this happens only during shutdown.
- `finally: try: sync_task.cancel()` — signals the background sync task to stop; wrapped in `try`/`finally` so that even if something below fails, the outer `finally: client.close()` still runs.
- `with suppress(asyncio.CancelledError): await sync_task` — waits for the sync task to actually finish stopping (which, per the docstring, may involve waiting for an in-progress sync cycle running in a background thread to notice a stop signal and exit cleanly), while quietly ignoring the `CancelledError` that cancelling a task normally raises — that error is the expected, normal outcome of cancellation here, not a real problem to propagate.
- `finally: client.close()` — releases the Qdrant client's resources (and its on-disk lock) no matter what happened above, guaranteeing it isn't left open on shutdown.

### Lines 99-109 — Building the `FastAPI` instance
```python
    app = FastAPI(
        title="Agentic RAG",
        description=(
            "A grounded football intelligence assistant: retrieval-augmented "
            "Q&A over an access-tiered document corpus, with prompt-injection, "
            "foul-language, and output-security screening. See "
            "docs/REQUIREMENTS.md for the full functional spec."
        ),
        version=importlib.metadata.version("agentic-rag"),
        lifespan=lifespan,
    )
```
- `app = FastAPI(...)` — constructs the actual FastAPI application object.
- `title="Agentic RAG"` — the name shown in the auto-generated API documentation.
- `description=(...)` — a longer description shown in the docs, summarizing the system's purpose (grounded football Q&A with security screening) and pointing readers at the full requirements document for details.
- `version=importlib.metadata.version("agentic-rag")` — pulls the real, currently-installed package version rather than hardcoding a version string that could drift out of sync with what's actually installed.
- `lifespan=lifespan` — wires in the startup/shutdown function defined above.

### Lines 110-111 — Mounting the routers
```python
    app.include_router(health_router)
    app.include_router(query_router)
```
- `app.include_router(health_router)` — registers the `/health` and `/health/ready` endpoints (from `api/routers/health.py`) onto the app.
- `app.include_router(query_router)` — registers the `/query` endpoint (from `api/routers/query.py`) onto the app.

### Lines 113-165 — Patching the OpenAPI schema for `/query`'s 422 response
```python
    default_openapi = app.openapi

    def _custom_openapi() -> dict:
        """Patch `POST /query`'s 422 response description onto FastAPI's
        own auto-generated schema, in place.
        ...
        """
        schema = default_openapi()
        query_422 = (
            schema.get("paths", {})
            .get("/query", {})
            .get("post", {})
            .get("responses", {})
            .get("422")
        )
        if query_422 is not None:
            query_422["description"] = QUERY_422_DESCRIPTION
        return schema

    app.openapi = _custom_openapi
    return app
```
- `default_openapi = app.openapi` — saves a reference to FastAPI's own, already-assigned method for generating the OpenAPI schema (the machine-readable description of the whole API, from which the interactive docs are built), before it gets replaced below.
- `def _custom_openapi() -> dict:` — defines a replacement function that will wrap the original schema-generation logic and modify its result.
- The docstring explains the underlying problem this function solves: `/query` can legitimately return a 422 status code in two different situations with two different response bodies (a validation failure with a `detail` array of field errors, or an unrecognized `user_tier` with a plain string `detail`), but FastAPI's automatic documentation only knows how to describe the first case out of the box. The obvious-seeming fix — adding `responses={422: {...}}` directly on the route decorator in `query.py` — doesn't work correctly: rather than adding to FastAPI's existing, auto-generated 422 entry, it entirely *replaces* it, which causes the reference to the `HTTPValidationError` schema component to disappear from the docs, and since nothing else on that route needs that reference anymore, FastAPI stops generating that whole schema component — so instead of accurately documenting both real 422 shapes, the docs end up documenting neither.
- It explains why this function delegates to `default_openapi()` (the method FastAPI itself originally assigned) instead of reimplementing OpenAPI-schema generation from scratch: `default_openapi()` already correctly handles caching (with logic to invalidate that cache if a router gets registered afterward, so the schema doesn't get stuck stale) and forwards all the other configuration passed to `FastAPI(...)` (like `contact`, `license_info`, `tags`, `servers`) into the underlying schema-generation call — reimplementing any of that by hand risks silently drifting out of sync with FastAPI's own behavior as the library evolves, which is exactly the kind of inaccuracy this whole patch exists to prevent.
- It explains that mutating the dictionary returned by `default_openapi()` in place also mutates the same object FastAPI has already cached internally on `app.openapi_schema` (because dictionaries in Python are shared references, not automatically copied) — so this patch effectively "sticks" across every future call for free, without `_custom_openapi()` needing to manage any caching of its own.
- It explains the `.get()` chain (instead of direct `[...]` indexing) is deliberate: if `/query`'s route ever changes shape such that FastAPI stops generating a 422 entry for it at all, this code silently does nothing instead of raising an error that would break `/openapi.json` — and therefore the entire interactive documentation page — for the whole application, not just this one endpoint's description.
- `schema = default_openapi()` — actually calls the original schema generator to get the full OpenAPI schema dictionary.
- `query_422 = schema.get("paths", {}).get("/query", {}).get("post", {}).get("responses", {}).get("422")` — safely navigates down through the nested schema structure (paths → `/query` → `post` → `responses` → `422`) using `.get()` at each step (which returns `None` instead of raising an error if a key is missing), to find the specific dictionary describing the 422 response for `POST /query`.
- `if query_422 is not None: query_422["description"] = QUERY_422_DESCRIPTION` — if that entry was actually found, overwrites its `"description"` field with the more accurate, two-shapes-explained text imported from `query.py`.
- `return schema` — returns the (possibly patched) schema.
- `app.openapi = _custom_openapi` — replaces the app's `openapi` method with this wrapped version, so every future call to generate the schema (including the one that powers `/openapi.json` and the interactive docs page) goes through the patch.
- `return app` — returns the fully assembled FastAPI application, ready to be run.
