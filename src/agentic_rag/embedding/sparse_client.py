from dataclasses import dataclass

from fastembed import SparseTextEmbedding

_model_cache: dict[str, SparseTextEmbedding] = {}


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


def _get_model(model_name: str) -> SparseTextEmbedding:
    """Sparse models load a small tokenizer/vocab bundle on first use;
    reuse one instance per model name instead of reloading it per call."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SparseTextEmbedding(model_name=model_name)
    return _model_cache[model_name]


def embed_sparse_texts(texts: list[str], model_name: str) -> list[SparseVector]:
    """Embed texts as BM25 sparse vectors, in order, via fastembed.

    Deterministic per text regardless of what else is in the batch (BM25
    here uses fixed term statistics, not corpus-fitted IDF) - required for
    a stable index and for these tests to be reproducible.
    """
    if not texts:
        return []

    model = _get_model(model_name)
    return [
        SparseVector(indices=list(result.indices), values=list(result.values))
        for result in model.embed(texts)
    ]
