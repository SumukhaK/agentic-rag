# `api/dependencies.py`

**Purpose:** This file defines small "getter" functions used by FastAPI's dependency injection system (a mechanism where a function declares what it needs as parameters, and the framework supplies those values automatically instead of the function reaching out and constructing or fetching them itself). Each function here pulls one shared resource — the app's settings, its Qdrant database client, its embedding cache, or its semantic cache — off of `request.app.state`, where `api/app.py`'s `lifespan` function stashed them once at startup. Routes then ask for these resources via `Depends(...)` instead of importing globals directly, which keeps the routes testable (a test can swap in different fake resources) and avoids the pitfalls of global mutable state.

## Line-by-line walkthrough

### Lines 1-6 — Imports
```python
from fastapi import Request
from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.orchestration.semantic_cache import SemanticCache
```
- `from fastapi import Request` — imports FastAPI's `Request` type, which represents the incoming HTTP request; each function below takes one as its parameter because that's how it reaches `request.app.state`, the place shared, app-wide objects are stored.
- `from qdrant_client import QdrantClient` — imports the type of the Qdrant database client, used here only as a type hint (for clarity and tooling support), not to construct anything.
- `from agentic_rag.config import Settings` — imports the `Settings` class (defined in `config.py`) purely as a type hint for the return value of `get_settings`.
- `from agentic_rag.embedding.cache import EmbeddingCache` — imports the `EmbeddingCache` type, used as a type hint for `get_embedding_cache`'s return value.
- `from agentic_rag.orchestration.semantic_cache import SemanticCache` — imports the `SemanticCache` type, used as a type hint for `get_semantic_cache`'s return value.

### Lines 9-10 — `get_settings`
```python
def get_settings(request: Request) -> Settings:
    return request.app.state.settings
```
- `def get_settings(request: Request) -> Settings:` — defines a function that FastAPI can call automatically (via `Depends(get_settings)` in a route) whenever a route needs the app's configuration.
- `return request.app.state.settings` — reads the `Settings` object off `app.state`, where it was placed once when the app started up (see `api/app.py`'s `lifespan` function), rather than constructing a new `Settings()` on every request.

### Lines 13-14 — `get_qdrant_client`
```python
def get_qdrant_client(request: Request) -> QdrantClient:
    return request.app.state.qdrant_client
```
- `def get_qdrant_client(request: Request) -> QdrantClient:` — defines the dependency function routes use to get access to the vector database client.
- `return request.app.state.qdrant_client` — returns the single, shared `QdrantClient` instance created once at startup. Because Qdrant's local/embedded mode locks its storage to a single process, this client must be reused rather than recreated per request.

### Lines 17-18 — `get_embedding_cache`
```python
def get_embedding_cache(request: Request) -> EmbeddingCache:
    return request.app.state.embedding_cache
```
- `def get_embedding_cache(request: Request) -> EmbeddingCache:` — defines the dependency function that supplies the shared embedding cache (a store that avoids recomputing embeddings for text that was already embedded before).
- `return request.app.state.embedding_cache` — returns the one `EmbeddingCache` created at startup; because caching only helps if it's shared across requests, this must not be created fresh each time.

### Lines 21-22 — `get_semantic_cache`
```python
def get_semantic_cache(request: Request) -> SemanticCache:
    return request.app.state.semantic_cache
```
- `def get_semantic_cache(request: Request) -> SemanticCache:` — defines the dependency function that supplies the shared semantic cache (a cache that can reuse a previous answer when a new query is semantically similar enough to one that was already answered).
- `return request.app.state.semantic_cache` — returns the one `SemanticCache` created at startup, for the same reason as `get_embedding_cache`: a per-request cache would never have anything cached in it yet, defeating its purpose.
