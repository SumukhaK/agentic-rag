from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from agentic_rag.api.routers.health import router as health_router
from agentic_rag.api.routers.query import QUERY_422_DESCRIPTION
from agentic_rag.api.routers.query import router as query_router
from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.orchestration.semantic_cache import SemanticCache


def _custom_openapi(app: FastAPI) -> dict:
    """Generate `app`'s OpenAPI schema, then correct `POST /query`'s 422
    response description in place.

    `/query` can return a 422 two different ways with two different
    bodies (see `QUERY_422_DESCRIPTION`), but FastAPI only knows how to
    auto-generate documentation for the request-validation-failure case.
    The obvious fix - passing `responses={422: {"description": ...}}` on
    the route decorator - doesn't merge with FastAPI's own auto-added 422
    entry, it *replaces* it: the `content`/schema reference to
    `HTTPValidationError` disappears, and since nothing else on this route
    ever requests that reference, FastAPI stops registering the
    `HTTPValidationError` component definition entirely - the response
    goes from "documents one of two real shapes" to "documents neither."
    Generating the correct schema first with `get_openapi()`, then only
    editing the description text of the entry it already built correctly,
    keeps the auto-generated `content`/component intact while still
    surfacing the second shape.

    Cached on `app.openapi_schema` the same way FastAPI's own default
    `openapi()` method caches it - regenerating this on every `/openapi.json`
    request would repeat the full schema walk for no benefit, since the
    schema doesn't change after startup.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["paths"]["/query"]["post"]["responses"]["422"]["description"] = (
        QUERY_422_DESCRIPTION
    )
    app.openapi_schema = schema
    return app.openapi_schema


def create_app(settings: Settings) -> FastAPI:
    """Build the Agentic RAG FastAPI app for `settings`.

    Takes `settings` explicitly rather than constructing a `Settings()`
    internally, matching this codebase's established explicit-parameter
    style (every function in `src/agentic_rag/` takes its config as
    arguments, not as an implicit global lookup) - it's also what lets
    tests point the app at an ephemeral `tmp_path` Qdrant/corpus instead
    of whatever `.env` happens to be on disk.

    The Qdrant client, embedding cache, and semantic cache are created
    once in `lifespan` and stored on `app.state`, not per-request: Qdrant's
    local/embedded mode (`qdrant_setup.get_client`) is a single-process,
    on-disk-locked client, and both caches are documented as scoped to a
    single process's lifetime (`EmbeddingCache`/`SemanticCache`
    docstrings) - creating a fresh one per request would silently defeat
    the whole point of caching. `ensure_collection` runs at startup so a
    schema mismatch fails fast when the app boots, not confusingly on the
    first query.
    """

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
        try:
            yield
        finally:
            client.close()

    app = FastAPI(
        title="Agentic RAG",
        description=(
            "A grounded football intelligence assistant: retrieval-augmented "
            "Q&A over an access-tiered document corpus, with prompt-injection, "
            "foul-language, and output-security screening. See "
            "docs/REQUIREMENTS.md for the full functional spec."
        ),
        version="0.1.0",  # tracks pyproject.toml's [project].version - keep both in sync
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(query_router)
    app.openapi = lambda: _custom_openapi(app)
    return app
