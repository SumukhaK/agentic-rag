import importlib.metadata
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_rag.api.routers.health import router as health_router
from agentic_rag.api.routers.query import QUERY_422_DESCRIPTION
from agentic_rag.api.routers.query import router as query_router
from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.orchestration.semantic_cache import SemanticCache


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
        version=importlib.metadata.version("agentic-rag"),
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(query_router)

    default_openapi = app.openapi

    def _custom_openapi() -> dict:
        """Patch `POST /query`'s 422 response description onto FastAPI's
        own auto-generated schema, in place.

        `/query` can return a 422 two different ways with two different
        bodies (see `QUERY_422_DESCRIPTION`), but FastAPI only knows how
        to auto-generate documentation for the request-validation-failure
        case. The obvious fix - passing `responses={422: {...}}` on the
        route decorator - doesn't merge with FastAPI's own auto-added 422
        entry, it *replaces* it: the `content`/schema reference to
        `HTTPValidationError` disappears, and since nothing else on the
        route ever requests that reference, FastAPI stops registering the
        `HTTPValidationError` component definition entirely - the response
        goes from "documents one of two real shapes" to "documents
        neither."

        Delegates to `default_openapi` - the bound method FastAPI itself
        assigned to `app.openapi` before this closure replaces it - rather
        than reimplementing schema generation by hand. `default_openapi()`
        already does the caching correctly (with route-version
        invalidation, so a router registered later doesn't get silently
        stuck out of a stale cached schema) and forwards every relevant
        `FastAPI(...)` constructor field (`contact`, `license_info`,
        `tags`, `servers`, ...) to `get_openapi()` - reimplementing either
        of those by hand risks quietly drifting from FastAPI's own
        behavior as it evolves, exactly the kind of inaccuracy this PR
        exists to catch. Mutating the dict `default_openapi()` returns
        also mutates what FastAPI cached on `app.openapi_schema` (dicts
        are references, not copies), so this patch survives across calls
        for free without this function needing to manage its own cache.

        Looked up via `.get()` chains rather than direct `[...]`
        indexing - if `/query`'s route shape ever changes such that
        FastAPI stops auto-generating a 422 entry for it, this silently
        skips the patch instead of raising and taking `/openapi.json`
        down for the entire app, not just this one endpoint.
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
