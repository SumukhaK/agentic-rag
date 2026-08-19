# `embedding/sparse_client.py`

**Purpose:** This file is responsible for computing "sparse embeddings" for text using a local library called `fastembed`, rather than a dense embedding server like Ollama. A sparse embedding (specifically BM25 here, a well-established keyword-scoring algorithm from search engines) represents a piece of text as a vector that is mostly zeros, with a few non-zero entries corresponding to important words/terms — this complements the "dense" embeddings computed elsewhere, which capture more general semantic meaning. Combining both kinds of vectors (a technique often called hybrid search) tends to give better search results than either alone: sparse vectors are strong at exact keyword matches, dense vectors are strong at conceptual similarity. This file loads the sparse model once, reuses it, and converts its output into the exact format Qdrant (the vector database used by this project) expects when storing vectors.

## Line-by-line walkthrough

### Lines 1-5 — Imports and module-level model cache
```python
from qdrant_client.models import SparseVector

from fastembed import SparseTextEmbedding

_model_cache: dict[str, SparseTextEmbedding] = {}
```
- `from qdrant_client.models import SparseVector` — imports Qdrant's own `SparseVector` type. Qdrant is the vector database this project uses to store and search embeddings, and it expects sparse vectors in this specific shape (a list of indices and a matching list of values) when they're inserted, so this file builds its results directly in that format rather than inventing its own.
- `from fastembed import SparseTextEmbedding` — imports the `fastembed` library's class for computing sparse (BM25-style) text embeddings locally, without needing a network call to an external service.
- `_model_cache: dict[str, SparseTextEmbedding] = {}` — a module-level (shared across the whole program) dictionary that will hold already-loaded sparse embedding models, keyed by model name. The leading underscore signals it's a private implementation detail not meant to be used from outside this file.

### Lines 8-13 — Custom error type
```python
class SparseEmbeddingError(Exception):
    """Raised when the sparse embedding model can't be loaded (e.g. an
    invalid model name, or no network on first use - fastembed downloads a
    small tokenizer/vocab bundle to the OS temp directory, not a stable
    cache location, so this can happen on any fresh environment) or fails
    to embed the given text."""
```
- `class SparseEmbeddingError(Exception):` — defines a dedicated exception type for anything that goes wrong in this file, mirroring the same pattern used for dense embeddings in `ollama_client.py`'s `EmbeddingError`. This lets calling code catch one consistent error type instead of dealing with whatever internal exception `fastembed` happens to raise.
- The docstring explains two distinct failure scenarios this error covers: the model failing to load in the first place (for example, because the model name is wrong, or because — on the very first use — `fastembed` needs to download a small tokenizer/vocabulary file and there's no network access to do so; the comment notes this download goes to the operating system's temporary directory rather than somewhere stable, so this failure can resurface on any newly-provisioned machine, not just truly offline ones), and the model successfully loading but then failing partway through actually embedding text.

### Lines 16-18 — `_get_model` signature and docstring
```python
def _get_model(model_name: str) -> SparseTextEmbedding:
    """Sparse models load a tokenizer/vocab bundle on first use; reuse one
    instance per model name instead of reloading it per call."""
```
- Defines a private helper function (leading underscore again marks it as internal to this file) that returns a loaded `SparseTextEmbedding` model for a given model name. The docstring states the reason it exists: since loading a sparse model involves loading a tokenizer/vocabulary bundle (a relatively slow, one-time setup cost), it's wasteful to reload it every time embeddings are needed — so the function caches loaded models and reuses them.

### Lines 19-26 — Loading and caching the model
```python
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = SparseTextEmbedding(model_name=model_name)
        except Exception as exc:
            raise SparseEmbeddingError(
                f"failed to load sparse embedding model '{model_name}': {exc}"
            ) from exc
    return _model_cache[model_name]
```
- `if model_name not in _model_cache:` — only does the (relatively expensive) work of loading a model if one hasn't already been loaded for this exact model name; otherwise, the function skips straight to returning the cached instance.
- `try: ... _model_cache[model_name] = SparseTextEmbedding(model_name=model_name)` — attempts to construct a new `SparseTextEmbedding` instance for the requested model name, which triggers `fastembed` to load (and, if necessary, download) the model's tokenizer/vocabulary bundle, and stores the result in the module-level cache dictionary so it can be reused by future calls.
- `except Exception as exc:` — catches any kind of failure during that loading process. A broad `Exception` catch is used here (rather than a specific error type) because `fastembed`/its dependencies could raise a variety of different error types depending on exactly what went wrong (bad model name, network failure, corrupted download, etc.), and this function's job is to normalize all of them into one project-specific error type.
- `raise SparseEmbeddingError(f"failed to load sparse embedding model '{model_name}': {exc}") from exc` — wraps whatever went wrong in the custom `SparseEmbeddingError`, including the model name and the original error message for context, and uses `from exc` to preserve the original exception as the documented cause for easier debugging.
- `return _model_cache[model_name]` — returns the model instance, whether it was just freshly loaded or had already been cached from an earlier call.

### Lines 29-38 — `embed_sparse_texts` signature and docstring
```python
def embed_sparse_texts(texts: list[str], model_name: str) -> list[SparseVector]:
    """Embed texts as BM25 sparse vectors, in order, via fastembed.

    Deterministic per text regardless of what else is in the batch (BM25
    here uses fixed term statistics, not corpus-fitted IDF) - required for
    a stable index and for these tests to be reproducible.

    Returns qdrant_client's own SparseVector type directly (not a custom
    one) since these vectors go straight into a Qdrant upsert.
    """
```
- Defines the main public function of this file: given a list of texts and the name of the sparse model to use, it returns one `SparseVector` per input text, in the same order.
- The docstring makes an important guarantee explicit: the sparse vector computed for a given piece of text doesn't depend on what other texts happen to be in the same batch. This is because the underlying BM25 algorithm, in this configuration, uses fixed statistics about how important each term is (rather than statistics computed dynamically from "IDF" — inverse document frequency, a measure of how rare/distinctive a word is across a whole collection of documents, which normally would change based on the batch or corpus it's computed over). This determinism matters for two practical reasons the docstring names: it keeps the search index stable (the same text always maps to the same vector, no matter when or with what else it's embedded), and it makes tests reproducible (the same input always produces the same output).
- The last paragraph explains a deliberate type choice: the function returns Qdrant's own `SparseVector` type directly, rather than defining a separate custom type for this file to use internally, because these vectors are headed straight into a Qdrant "upsert" (a combined insert-or-update operation) — there's no benefit to an extra intermediate type that would just need to be converted again later.

### Lines 39-40 — Early exit for empty input
```python
    if not texts:
        return []
```
- If given an empty list of texts, returns an empty list immediately without loading a model or doing any embedding work — avoids unnecessary work and possibly loading a model that never ends up being used.

### Lines 42-46 — Loading the model and computing embeddings
```python
    model = _get_model(model_name)
    try:
        results = list(model.embed(texts))
    except Exception as exc:
        raise SparseEmbeddingError(f"failed to embed text: {exc}") from exc
```
- `model = _get_model(model_name)` — gets the (possibly cached) loaded model instance using the helper function defined above.
- `try: results = list(model.embed(texts))` — calls `fastembed`'s `embed` method on the list of texts to compute their sparse embeddings. `model.embed(texts)` returns a lazy generator/iterator (something that produces results one at a time as needed, rather than all at once) rather than a plain list, so wrapping it in `list(...)` forces all results to actually be computed now and stored in a normal list, so any error during embedding surfaces here inside this `try` block rather than later, and so the results can be safely indexed/iterated multiple times below.
- `except Exception as exc:` — again uses a broad catch, for the same reason as in `_get_model`: `fastembed`'s embedding step could fail in various ways, and this function's job is to translate any of them into the project's own `SparseEmbeddingError`.
- `raise SparseEmbeddingError(f"failed to embed text: {exc}") from exc` — raises the custom error with context about what failed, preserving the original exception via `from exc`.

### Lines 48-51 — Converting results into Qdrant's `SparseVector` format
```python
    return [
        SparseVector(indices=list(result.indices), values=list(result.values))
        for result in results
    ]
```
- Builds and returns the final list of `SparseVector` objects, one per input text, preserving their original order (since `results` came from iterating `texts` in order, and this list comprehension iterates `results` in that same order).
- For each `result` (fastembed's own internal sparse-embedding object), `SparseVector(indices=list(result.indices), values=list(result.values))` constructs Qdrant's expected representation: `indices` are the positions (within the model's vocabulary) of the non-zero entries, and `values` are the corresponding weights/scores at those positions — together these two parallel lists represent the "mostly zero" sparse vector compactly, without storing all the actual zero entries. Wrapping `result.indices` and `result.values` in `list(...)` converts whatever array-like type `fastembed` returns them as (for example, a NumPy array) into plain Python lists, which is the type `SparseVector` expects.
