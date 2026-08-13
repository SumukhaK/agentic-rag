from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from agentic_rag.retrieval.search import SearchCandidate

_model_cache: dict[str, TextCrossEncoder] = {}


class RerankError(Exception):
    """Raised when the reranker model can't be loaded, or fails to score
    the given candidates."""


def _get_model(model_name: str) -> TextCrossEncoder:
    """Reranker models load ONNX weights on first use; reuse one instance
    per model name instead of reloading it on every call, same rationale
    as sparse_client.py's model cache."""
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = TextCrossEncoder(model_name=model_name)
        except Exception as exc:
            raise RerankError(
                f"failed to load reranker model '{model_name}': {exc}"
            ) from exc
    return _model_cache[model_name]


def rerank(
    query: str, candidates: list[SearchCandidate], model_name: str, top_k: int
) -> list[SearchCandidate]:
    """Rerank `candidates` by relevance to `query` via a local cross-encoder,
    returning the best `top_k`.

    Each returned candidate's `score` is replaced with the cross-encoder's
    own relevance score - a more accurate signal than the fused hybrid
    search score for the chunks that actually reach generation.
    """
    if not candidates:
        return []

    model = _get_model(model_name)
    try:
        scores = list(model.rerank(query, [candidate.text for candidate in candidates]))
    except Exception as exc:
        raise RerankError(f"failed to rerank candidates: {exc}") from exc

    if len(scores) != len(candidates):
        raise RerankError(
            f"reranker returned {len(scores)} score(s) for {len(candidates)} candidate(s)"
        )

    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [
        replace(candidate, score=score) for score, candidate in ranked[:top_k]
    ]
