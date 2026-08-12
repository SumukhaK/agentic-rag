from qdrant_client.models import SparseVector

from fastembed import SparseTextEmbedding

_model_cache: dict[str, SparseTextEmbedding] = {}


class SparseEmbeddingError(Exception):
    """Raised when the sparse embedding model can't be loaded (e.g. an
    invalid model name, or no network on first use - fastembed downloads a
    small tokenizer/vocab bundle to the OS temp directory, not a stable
    cache location, so this can happen on any fresh environment) or fails
    to embed the given text."""


def _get_model(model_name: str) -> SparseTextEmbedding:
    """Sparse models load a tokenizer/vocab bundle on first use; reuse one
    instance per model name instead of reloading it per call."""
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = SparseTextEmbedding(model_name=model_name)
        except Exception as exc:
            raise SparseEmbeddingError(
                f"failed to load sparse embedding model '{model_name}': {exc}"
            ) from exc
    return _model_cache[model_name]


def embed_sparse_texts(texts: list[str], model_name: str) -> list[SparseVector]:
    """Embed texts as BM25 sparse vectors, in order, via fastembed.

    Deterministic per text regardless of what else is in the batch (BM25
    here uses fixed term statistics, not corpus-fitted IDF) - required for
    a stable index and for these tests to be reproducible.

    Returns qdrant_client's own SparseVector type directly (not a custom
    one) since these vectors go straight into a Qdrant upsert.
    """
    if not texts:
        return []

    model = _get_model(model_name)
    try:
        results = list(model.embed(texts))
    except Exception as exc:
        raise SparseEmbeddingError(f"failed to embed text: {exc}") from exc

    return [
        SparseVector(indices=list(result.indices), values=list(result.values))
        for result in results
    ]
