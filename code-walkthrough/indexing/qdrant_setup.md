# `indexing/qdrant_setup.py`

**Purpose:** This file is responsible for creating and connecting to the Qdrant collection that stores this system's searchable document chunks. Qdrant is a "vector database" — a database specialized for storing numeric vectors (lists of numbers representing meaning) and finding the ones most similar to a query vector, which is the core mechanism behind semantic search. This file sets up the local, on-disk Qdrant instance and defines the shape of the collection (what kinds of vectors it stores and how similarity between them is measured) before any documents are indexed into it. Getting this schema right up front matters because some choices, like adding a new named vector field, can't be changed later without recreating the whole collection.

## Line-by-line walkthrough

### Lines 1-7 — Imports and constants
```python
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
```
- `from pathlib import Path` — imports `Path`, used later purely as a type hint so the function signature can accept either a plain string or a proper filesystem path object for the storage location.
- `from qdrant_client import QdrantClient` — imports the official Qdrant Python client class, which this module wraps to create a connection.
- `from qdrant_client.models import Distance, SparseVectorParams, VectorParams` — imports the data classes Qdrant uses to describe a collection's schema: `Distance` (which similarity metric to use, e.g. cosine), `VectorParams` (the size/dimensionality and distance metric for a "dense" vector field), and `SparseVectorParams` (the configuration for a "sparse" vector field, used for keyword-style search rather than dense semantic embeddings).
- `DENSE_VECTOR_NAME = "dense"` and `SPARSE_VECTOR_NAME = "sparse"` — module-level constants naming the two vector fields that every point (a stored chunk) in the collection will have. Defining these as named constants (rather than repeating the string literals `"dense"`/`"sparse"` everywhere) means other modules that need to reference the same field names — such as `upsert.py` — import these constants instead of risking a typo causing a silent mismatch.

### Lines 10-13 — Custom exception type
```python
class CollectionSchemaMismatchError(Exception):
    """Raised when an existing collection's dense vector size doesn't match
    what's being requested - e.g. the embedding model/dimensions changed
    without the collection being migrated."""
```
- `class CollectionSchemaMismatchError(Exception):` — a dedicated exception used to fail loudly rather than silently misbehave if the code that ensures a collection exists finds that collection already configured with a different vector size than what's currently requested. This typically means the embedding model producing the vectors changed (different models output vectors of different lengths) but nobody migrated the existing collection to match, which would otherwise cause confusing downstream errors or silently broken search results.

### Lines 16-22 — `get_client`: connecting to Qdrant
```python
def get_client(storage_path: str | Path) -> QdrantClient:
    """Local/embedded Qdrant client: on-disk storage at `storage_path`, no
    server process. Docker isn't available in this dev environment - see
    docs/REQUIREMENTS.md §5. Swappable for a real server later by passing
    a `url=` instead of `path=` here.
    """
    return QdrantClient(path=str(storage_path))
```
- `def get_client(storage_path: str | Path) -> QdrantClient:` — defines a function that returns a connected `QdrantClient`. It accepts either a string or a `Path` for flexibility, and returns the client object callers use for all subsequent Qdrant operations.
- The docstring explains this uses Qdrant's *embedded* mode: rather than connecting to a separately-running Qdrant server process (which is the more typical production setup, often run via Docker), the client itself manages an on-disk database at `storage_path`. The comment notes this choice was made because Docker isn't available in the development environment this project targets, with a pointer to further requirements documentation. It also notes the design is intentionally easy to swap later — passing a `url=` argument instead of `path=` when constructing `QdrantClient` would switch to talking to a real remote/local server instead, without needing to change how the rest of the codebase calls `get_client`.
- `return QdrantClient(path=str(storage_path))` — constructs and returns the client, converting `storage_path` to a plain string (since `QdrantClient` expects a string path) regardless of whether a `Path` object or string was passed in.

### Lines 25-57 — `ensure_collection`: signature and docstring
```python
def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create `collection_name` if it doesn't already exist. Idempotent.

    Created with a named dense vector ("dense") *and* a named sparse
    vector ("sparse") from the start, even though sparse vectors aren't
    populated until native hybrid search (the next Phase 2 item) is wired
    in - Qdrant can't add a sparse vector field to an existing collection
    after creation, only recreate it, so setting the schema up right here
    avoids a full reindex later.

    Qdrant indexes dense vectors with HNSW by default - there's no
    alternative index to opt into, so this already satisfies the HNSW
    requirement with no manual tuning needed.

    Distance is fixed to cosine, not an overridable parameter - it's the
    metric `nomic-embed-text` (and this codebase's own semantic-cache
    similarity check) is built around, not an environment-varying tunable
    like `vector_size`. It was briefly a defaulted parameter with zero
    real call sites ever passing a non-default value; self-review during
    a hygiene audit flagged that as unused, speculative flexibility this
    project's own conventions rule out, and as a real gap besides - the
    mismatch check below only ever validated `vector_size`, so a caller
    that did pass a different distance against an existing collection
    would have been silently accepted instead of raising.

    Raises CollectionSchemaMismatchError if the collection already exists
    with a different dense vector size than requested, rather than
    silently leaving the mismatched collection in place.
    """
```
- `def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:` — takes an already-connected client, the name of the collection to create/check, and the dimensionality (`vector_size`) that the dense vectors must have (this must match whatever embedding model produces the vectors, since embedding models output a fixed-length vector). Returns nothing (`None`) — it's called purely for its side effect of guaranteeing the collection exists correctly.
- The docstring's first line states the core behavior: create the collection if missing, and do nothing if it's already there — this makes the function "idempotent," meaning calling it repeatedly has the same effect as calling it once, so it's safe to call on every startup without worrying about duplicate-creation errors.
- The second paragraph explains why the collection is created with *both* a dense vector field and a sparse vector field even though, at the time this code was written, only dense vectors are actually populated with data. Sparse vectors represent something like traditional keyword-based (rather than semantic) matching, planned for a later "hybrid search" feature. The key constraint driving this decision: Qdrant does not allow adding a new named vector field to a collection that already exists — the only way to add one later would be to recreate the entire collection from scratch, which means re-embedding and re-inserting every previously indexed document. Setting up the field now, even unused, avoids that expensive migration later.
- The third paragraph clarifies that Qdrant's default indexing algorithm for dense vectors is HNSW (Hierarchical Navigable Small World — a graph-based structure that makes approximate nearest-neighbor search fast even over large vector collections). Because this is Qdrant's built-in default with no alternative to choose from, the code doesn't need to explicitly configure or tune anything to satisfy a requirement for HNSW indexing — it comes for free.
- The fourth paragraph explains why the similarity metric (`Distance`) is hardcoded to cosine similarity rather than being an argument callers can override. Cosine distance is the metric the specific embedding model in use (`nomic-embed-text`) was designed around, and it's also what the codebase's own semantic-cache lookups (elsewhere in the system) rely on for consistency — so unlike `vector_size` (which legitimately varies if the embedding model changes), the distance metric isn't something that should vary per environment. The docstring also documents the history behind this: it used to be a parameter with a default value, but a code-hygiene review found that literally no caller ever passed a non-default value, meaning the flexibility was pure unused complexity. Worse, the mismatch-detection logic further down only ever checked `vector_size`, not distance — so if some future caller *had* passed a different distance against an already-existing collection, the mismatch wouldn't have been caught, silently creating an inconsistent setup. Removing the parameter entirely closes that gap.
- The final paragraph documents the explicit contract: if the collection already exists but with a different `vector_size` than what's being requested now, the function raises `CollectionSchemaMismatchError` rather than proceeding as if nothing were wrong — this is a "loud failure," consistent with the project's broader philosophy of not silently ignoring data-quality problems.

### Lines 58-67 — Checking for an existing collection and validating its schema
```python
    if client.collection_exists(collection_name):
        existing = client.get_collection(collection_name).config.params.vectors[
            DENSE_VECTOR_NAME
        ]
        if existing.size != vector_size:
            raise CollectionSchemaMismatchError(
                f"'{collection_name}' already exists with dense vector size "
                f"{existing.size}, but {vector_size} was requested"
            )
        return
```
- `if client.collection_exists(collection_name):` — asks Qdrant whether a collection with this name already exists, which is what makes this function idempotent — the creation logic is skipped entirely if there's nothing to do.
- `existing = client.get_collection(collection_name).config.params.vectors[DENSE_VECTOR_NAME]` — fetches the existing collection's full configuration and drills down into its vector configuration for the "dense" field specifically (using the `DENSE_VECTOR_NAME` constant defined earlier, rather than a hardcoded string), pulling out the settings (including its configured size) for comparison.
- `if existing.size != vector_size:` — compares the dimensionality the existing collection was created with against the dimensionality currently being requested.
- `raise CollectionSchemaMismatchError(...)` — if they don't match, raises the custom exception with a message that includes both the existing size and the requested size, so whoever sees the error immediately understands the nature of the mismatch without needing to dig further.
- `return` — if the collection exists and the size matches, the function simply returns early, having verified everything is already correct; there's nothing left to do.

### Lines 69-75 — Creating the collection
```python
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
    )
```
- `client.create_collection(collection_name=collection_name, ...)` — this line (and its arguments) only run if the earlier `if client.collection_exists(...)` block didn't already return, meaning the collection genuinely doesn't exist yet and needs to be created.
- `vectors_config={DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE)}` — defines the collection's dense vector field, named via the `DENSE_VECTOR_NAME` constant ("dense"), configured with the requested `vector_size` (must match the embedding model's output length) and a fixed `Distance.COSINE` similarity metric, per the reasoning in the docstring above.
- `sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()}` — defines the collection's sparse vector field, named via `SPARSE_VECTOR_NAME` ("sparse"), using Qdrant's default sparse vector settings (no dense-vector-style size is needed since sparse vectors are variable-length by nature). As explained in the docstring, this field is set up now even though nothing populates it yet, because it cannot be added retroactively without recreating the collection.
