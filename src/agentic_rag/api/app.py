from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_rag.api.routers.health import router as health_router
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

    app = FastAPI(title="Agentic RAG", lifespan=lifespan)
    app.include_router(health_router)
    return app
