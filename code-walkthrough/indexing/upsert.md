# `indexing/upsert.py`

**Purpose:** This file is responsible for taking a single ingested document (already split into text chunks by an earlier pipeline stage) and getting it into Qdrant as searchable vectors — a process commonly called "upserting" (update-or-insert). It handles turning each chunk's text into both a dense vector (a semantic embedding capturing meaning) and a sparse vector (for keyword-style matching), assigning each chunk a stable ID so re-indexing the same document doesn't create duplicates, and carefully ordering the delete/insert steps so that re-indexing a document that changed (or failed partway through) never leaves the search index in a broken or inconsistent state.

## Line-by-line walkthrough

### Lines 1-10 — Imports
```python
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from agentic_rag.embedding.cache import EmbeddingCache, embed_with_cache
from agentic_rag.embedding.ollama_client import embed_texts
from agentic_rag.embedding.sparse_client import embed_sparse_texts
from agentic_rag.indexing.qdrant_setup import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from agentic_rag.ingestion.pipeline import IngestedDocument
```
- `import uuid` — the standard library module used to generate deterministic, stable identifiers for each indexed chunk (explained further below).
- `from qdrant_client import QdrantClient` — the Qdrant client class, used here as a type hint and to call operations like `delete` and `upsert` against the database.
- `from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct` — imports Qdrant's data types: `PointStruct` represents one storable item (a vector plus metadata) in the collection; `Filter`, `FieldCondition`, and `MatchValue` are used together to describe a "find/delete everything matching this condition" query, used here to select points belonging to a specific document.
- `from agentic_rag.embedding.cache import EmbeddingCache, embed_with_cache` — brings in the caching layer for embeddings: `EmbeddingCache` is the cache object type, and `embed_with_cache` is a helper that wraps an embedding function so identical text doesn't get re-embedded (an expensive operation) more than once.
- `from agentic_rag.embedding.ollama_client import embed_texts` — the function that actually calls Ollama to turn text chunks into dense (semantic) embedding vectors.
- `from agentic_rag.embedding.sparse_client import embed_sparse_texts` — the function that turns text chunks into sparse vectors (keyword-style representations), used for the sparse half of hybrid search.
- `from agentic_rag.indexing.qdrant_setup import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME` — imports the shared field-name constants ("dense"/"sparse") defined in `qdrant_setup.py`, so this file references the exact same vector field names used when the collection schema was created, rather than risking a mismatched hardcoded string.
- `from agentic_rag.ingestion.pipeline import IngestedDocument` — imports the data type representing a document that has already been loaded and split into chunks upstream, which is what this module consumes as input.

### Lines 12-14 — Fixed UUID namespace
```python
# Fixed, arbitrary namespace for deterministic point IDs - only needs to be
# stable across runs of this codebase, not globally unique.
_POINT_ID_NAMESPACE = uuid.UUID("5c1e6f5a-6b8e-4b8a-9b0a-8f1d2c3e4f5a")
```
- The comment explains the purpose of this constant: it's used as a fixed "namespace" input to a deterministic UUID-generation function, so the same inputs always produce the same output ID. It only needs to stay consistent within this codebase (so that re-running indexing produces the same IDs each time) — it doesn't need to be universally unique in the way a randomly-generated UUID would, since its only job is to seed deterministic generation, not to identify anything by itself.
- `_POINT_ID_NAMESPACE = uuid.UUID("5c1e6f5a-6b8e-4b8a-9b0a-8f1d2c3e4f5a")` — defines this fixed namespace as a module-level constant (the leading underscore signals it's private to this module), a specific hardcoded UUID value chosen arbitrarily but fixed forever once chosen, since changing it later would change every point ID generated from it.

### Lines 17-22 — `_point_id`: deterministic chunk IDs
```python
def _point_id(relative_path: str, chunk_index: int) -> str:
    """Deterministic per (document, chunk): re-indexing the same chunk
    always produces the same point ID, making index_document() safe to
    retry - running it twice for the same document converges to the same
    set of points instead of accumulating duplicates."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{relative_path}::{chunk_index}"))
```
- `def _point_id(relative_path: str, chunk_index: int) -> str:` — a private helper function (leading underscore) that computes the unique ID Qdrant will use for one specific chunk of one specific document, based on the document's path and the chunk's position within it.
- The docstring explains why determinism matters here: if the ID were random each time, re-indexing the same document twice would create two separate sets of points for the same content (duplicates) instead of the second run simply overwriting the first. By deriving the ID purely from `(relative_path, chunk_index)`, the same chunk always maps to the same ID, so indexing the same document repeatedly (e.g. because a sync job re-processes it, or a previous attempt failed partway through) safely converges on one consistent set of points rather than piling up duplicates.
- `return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{relative_path}::{chunk_index}"))` — uses `uuid.uuid5`, which generates a UUID deterministically from a namespace and a name string (as opposed to `uuid.uuid4`, which is random). The "name" fed in is `f"{relative_path}::{chunk_index}"`, combining the document's path and the chunk's index into one string so different chunks of the same document (or same-index chunks of different documents) never collide. The result is converted to a string since that's the ID format Qdrant point structs expect.

### Lines 25-28 — `_path_filter`: building a Qdrant filter by document path
```python
def _path_filter(relative_path: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="relative_path", match=MatchValue(value=relative_path))]
    )
```
- `def _path_filter(relative_path: str) -> Filter:` — another private helper, building a reusable Qdrant `Filter` object that matches all points belonging to one document, identified by its `relative_path`.
- `return Filter(must=[FieldCondition(key="relative_path", match=MatchValue(value=relative_path))])` — constructs a filter requiring (`must`) that the point's payload field named `"relative_path"` exactly equals (`MatchValue`) the given path. This is what lets the module find or delete "every chunk belonging to this document" without needing to know the individual chunk IDs.

### Lines 31-33 — `delete_document`: removing all points for a document
```python
def delete_document(client: QdrantClient, collection_name: str, relative_path: str) -> None:
    """Remove every indexed point for `relative_path`. A no-op if none exist."""
    client.delete(collection_name=collection_name, points_selector=_path_filter(relative_path))
```
- `def delete_document(client: QdrantClient, collection_name: str, relative_path: str) -> None:` — a public function that deletes all previously indexed chunks belonging to a given document path from the specified collection.
- The docstring clarifies that calling this on a document with nothing currently indexed is harmless — it simply does nothing (a "no-op"), so callers don't need to check existence first.
- `client.delete(collection_name=collection_name, points_selector=_path_filter(relative_path))` — calls Qdrant's delete operation, using the filter built by `_path_filter` (rather than a list of specific point IDs) as the selector, so it deletes by matching the `relative_path` payload field regardless of how many chunks that document currently has indexed.

### Lines 36-70 — `index_document`: signature and docstring
```python
def index_document(
    client: QdrantClient,
    collection_name: str,
    document: IngestedDocument,
    *,
    embedding_model: str,
    ollama_base_url: str,
    sparse_model: str,
    embedding_timeout_seconds: int,
    embedding_cache: EmbeddingCache,
) -> None:
    """Embed every chunk of `document` (dense + sparse) and upsert it as a
    Qdrant point, with the payload citation/access-control needs at query
    time.

    Embedding happens *before* anything is deleted from the index. If
    embed_texts()/embed_sparse_texts() raises (a transient Ollama
    hiccup, a timeout under load), a document that was already indexed
    stays exactly as it was rather than silently vanishing from search
    results - deleting first and only then embedding would leave a
    window where the old points are gone and nothing has replaced them
    yet.

    Existing points for this document's relative_path are deleted only
    once the new chunk set is fully embedded and length-checked, then the
    current chunk set is inserted fresh. This is what keeps an edit from
    leaving orphaned points behind: if a document shrinks from 5 chunks to
    3, only upserting the new 3 (without first clearing what was there)
    would leave chunks 3-4 stale and searchable forever.

    `embedding_cache` is shared across calls by the caller (e.g. one
    instance per sync cycle, not one per document) - that's what lets
    identical chunk text repeated across different documents (boilerplate,
    headers, disclaimers) skip re-embedding.
    """
```
- `def index_document(client: QdrantClient, collection_name: str, document: IngestedDocument, *, embedding_model: str, ollama_base_url: str, sparse_model: str, embedding_timeout_seconds: int, embedding_cache: EmbeddingCache) -> None:` — this is the main entry point of the file. It takes the Qdrant client and target collection, the already-chunked `document` to index, and a set of keyword-only configuration arguments: which embedding model to use for dense vectors, where the Ollama server lives, which model to use for sparse vectors, how long to wait for embedding calls before timing out, and a shared cache object to avoid redundant embedding work. Making all the configuration arguments keyword-only (via the `*`) forces every call site to name them explicitly, avoiding mix-ups between similarly-typed parameters like the two model names.
- The docstring's opening line summarizes the job: embed every chunk both ways (dense and sparse) and store it in Qdrant along with payload data needed later — both to show citations (which document/chunk an answer came from) and to enforce access control (e.g. filtering search results by an `access_tier`) when queries run.
- The second paragraph explains a critical ordering decision: embedding happens *before* any deletion of old data. If the embedding step fails partway through (Ollama being temporarily unreachable, or timing out under heavy load), the function raises before touching the existing index, so a document that was previously indexed remains searchable exactly as before rather than disappearing. The alternative order — deleting old points first, then trying to embed and insert new ones — would create a window where, if the embedding step then failed, the document would have no points in the index at all, silently vanishing from search results until someone noticed and re-ran indexing.
- The third paragraph explains why deletion still eventually happens, and specifically *when*: only after the new chunk set has been fully embedded and had its lengths checked (see the mismatch check below), the old points get deleted and replaced with the new set. This ordering solves a different problem: if a document is edited and shrinks (say from 5 chunks down to 3), simply upserting the 3 new chunks without first deleting everything old would leave the old chunks 4 and 5 sitting in the index forever ("orphaned"), since nothing would overwrite or remove them — they'd keep showing up in search results even though that content no longer exists in the source document.
- The fourth paragraph documents an expectation about how `embedding_cache` should be used by callers: it's meant to be a single shared instance reused across many calls to `index_document` (e.g., one cache per indexing run across a whole sync cycle) rather than a fresh cache created per document. This is what allows text that repeats across multiple different documents — common boilerplate, shared headers, disclaimers — to be embedded once and reused, rather than paying the cost of re-embedding identical text every time it's encountered in a different document.

### Lines 71-73 — Handling a document with no chunks
```python
    if not document.chunks:
        delete_document(client, collection_name, document.relative_path)
        return
```
- `if not document.chunks:` — checks whether the document has no chunks at all (e.g. it was emptied out, deleted, or filtered down to nothing upstream).
- `delete_document(client, collection_name, document.relative_path)` — if so, the correct action is simply to remove anything previously indexed for this document, since there's no new content to replace it with.
- `return` — exits early; there's nothing further to embed or insert.

### Lines 75-86 — Embedding chunk text into dense vectors (with caching)
```python
    texts = [chunk.text for chunk in document.chunks]
    dense_vectors = embed_with_cache(
        texts,
        model=embedding_model,
        cache=embedding_cache,
        embed_fn=lambda batch: embed_texts(
            batch,
            model=embedding_model,
            base_url=ollama_base_url,
            timeout=embedding_timeout_seconds,
        ),
    )
```
- `texts = [chunk.text for chunk in document.chunks]` — extracts just the raw text string from each chunk into a plain list, since that's what the embedding functions operate on (not the full chunk objects with their metadata).
- `dense_vectors = embed_with_cache(texts, model=embedding_model, cache=embedding_cache, embed_fn=lambda batch: embed_texts(batch, model=embedding_model, base_url=ollama_base_url, timeout=embedding_timeout_seconds))` — calls the caching wrapper to get a dense embedding vector for each text. `embed_with_cache` presumably checks the shared `embedding_cache` first for any text it's already seen (for this `model`), and only calls the supplied `embed_fn` for whatever wasn't already cached. The `embed_fn` here is a small anonymous function (`lambda`) that, when invoked with a batch of texts needing embedding, calls the real `embed_texts` function against Ollama with the configured model, base URL, and timeout. Wrapping the call in a lambda lets `embed_with_cache` control exactly which texts get passed through to the real embedding call (only the cache misses), while still letting this function supply the fixed configuration values.

### Lines 87-92 — Embedding chunk text into sparse vectors (with caching)
```python
    sparse_vectors = embed_with_cache(
        texts,
        model=sparse_model,
        cache=embedding_cache,
        embed_fn=lambda batch: embed_sparse_texts(batch, model_name=sparse_model),
    )
```
- `sparse_vectors = embed_with_cache(texts, model=sparse_model, cache=embedding_cache, embed_fn=lambda batch: embed_sparse_texts(batch, model_name=sparse_model))` — mirrors the dense embedding call above, but for sparse vectors: it reuses the same `texts` and the same shared `embedding_cache`, but keys/caches under `sparse_model` instead of `embedding_model` (since dense and sparse embeddings for the same text are different values and must be cached separately), and its `embed_fn` lambda calls `embed_sparse_texts` instead of `embed_texts`.

### Lines 94-104 — Guarding against a mismatched embedding count
```python
    if len(dense_vectors) != len(document.chunks) or len(sparse_vectors) != len(
        document.chunks
    ):
        # zip() below would otherwise silently truncate to the shortest
        # list instead of erroring on a malformed/incomplete response from
        # either embedding client.
        raise ValueError(
            f"embedding count mismatch for '{document.relative_path}': "
            f"{len(document.chunks)} chunks, {len(dense_vectors)} dense vectors, "
            f"{len(sparse_vectors)} sparse vectors"
        )
```
- `if len(dense_vectors) != len(document.chunks) or len(sparse_vectors) != len(document.chunks):` — checks that both the dense and sparse embedding calls returned exactly one vector per chunk, no more and no fewer.
- The comment explains why this check exists: the code below uses `zip()` to pair up chunks with their dense and sparse vectors, and `zip()` silently stops at the length of the *shortest* input list rather than raising an error if the lists are different lengths. Without this explicit check, a bug or partial failure in one of the embedding clients that returned too few (or too many) vectors would silently produce a smaller, wrong set of points instead of surfacing the problem — violating the project's principle that data quality failures should be loud, not silently swallowed.
- `raise ValueError(f"embedding count mismatch for '{document.relative_path}': {len(document.chunks)} chunks, {len(dense_vectors)} dense vectors, {len(sparse_vectors)} sparse vectors")` — raises an error with a message specifically naming the document and the exact counts involved, making it immediately clear what went wrong and for which document if this ever happens.

### Lines 106-118 — Building the Qdrant point structures
```python
    points = [
        PointStruct(
            id=_point_id(document.relative_path, chunk.index),
            vector={DENSE_VECTOR_NAME: dense, SPARSE_VECTOR_NAME: sparse},
            payload={
                "relative_path": document.relative_path,
                "chunk_index": chunk.index,
                "text": chunk.text,
                "access_tier": document.access_tier,
            },
        )
        for chunk, dense, sparse in zip(document.chunks, dense_vectors, sparse_vectors)
    ]
```
- `points = [...]` — builds a list of `PointStruct` objects, one per chunk, ready to be inserted into Qdrant.
- `for chunk, dense, sparse in zip(document.chunks, dense_vectors, sparse_vectors)` — iterates over the three lists in lockstep, pairing each chunk with its corresponding dense vector and sparse vector (this is safe now precisely because the length check above already guaranteed all three lists are the same length).
- `id=_point_id(document.relative_path, chunk.index)` — assigns the deterministic ID computed earlier, ensuring re-indexing overwrites the same point rather than creating a duplicate.
- `vector={DENSE_VECTOR_NAME: dense, SPARSE_VECTOR_NAME: sparse}` — stores both vector types on the point under their respective named fields (using the shared constants imported from `qdrant_setup.py`), matching the collection schema that was created with both a "dense" and a "sparse" named vector.
- `payload={"relative_path": document.relative_path, "chunk_index": chunk.index, "text": chunk.text, "access_tier": document.access_tier}` — attaches metadata alongside the vectors: `relative_path` and `chunk_index` identify exactly where this chunk came from (used for citations and for the delete-by-path filter defined earlier), `text` stores the original chunk content so it can be returned directly in search results without a separate lookup, and `access_tier` carries whatever access-control classification the source document has, so queries can later be filtered to only return chunks a given user is permitted to see.

### Lines 120-121 — Replacing old points with the new set
```python
    delete_document(client, collection_name, document.relative_path)
    client.upsert(collection_name=collection_name, points=points)
```
- `delete_document(client, collection_name, document.relative_path)` — only now, after embedding succeeded and the counts were validated, are the old points for this document removed. As explained in the docstring, this ordering means a failure earlier in the function (during embedding) never reaches this line, so the old index state is preserved on failure and only replaced once the new state is fully ready.
- `client.upsert(collection_name=collection_name, points=points)` — inserts the freshly built points into the collection. Because each point's ID is deterministic and the old points were just deleted, this results in exactly the current chunk set being present afterward — no leftover stale chunks from a previous, differently-sized version of the document, and no duplicates from repeated indexing runs.
