# `retrieval/rerank.py`

**Purpose:** After the hybrid search step (in `search.py`) pulls back a rough list of candidate text chunks using fast vector similarity, that ranking is only an approximation of true relevance — it's good at finding plausible matches quickly, but not great at fine-grained judgments about which chunk best answers the specific query. This file adds a second, more accurate pass: a "cross-encoder" reranker, which is a small local machine learning model that looks directly at the query and each candidate's text *together* (rather than comparing pre-computed vectors) and produces a much more trustworthy relevance score. This file is responsible for loading that reranker model, running it over a batch of candidates, and returning only the best `top_k` of them with updated, more accurate scores — so that only the most relevant chunks make it into the final answer-generation step.

## Line-by-line walkthrough

### Lines 1-7 — Imports and the model cache
```python
from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from agentic_rag.retrieval.search import SearchCandidate

_model_cache: dict[str, TextCrossEncoder] = {}
```
- `from dataclasses import replace` — imports the `replace` helper, which creates a copy of a frozen (immutable) dataclass instance with one or more fields swapped out. This is used later to produce a new `SearchCandidate` that has the same text/path/tier but an updated `score`, without needing to mutate the original object (which isn't allowed since it's frozen).
- `from fastembed.rerank.cross_encoder import TextCrossEncoder` — imports the actual cross-encoder reranking model class from the `fastembed` library. A cross-encoder is a model that takes a (query, document) pair as joint input and outputs a single relevance score, which tends to be more accurate than comparing separately-computed embedding vectors, at the cost of being slower to run on many candidates.
- `from agentic_rag.retrieval.search import SearchCandidate` — imports the `SearchCandidate` data structure defined in `search.py`, since this file both accepts and returns lists of these objects.
- `_model_cache: dict[str, TextCrossEncoder] = {}` — a module-level (i.e., shared across all calls within the process) dictionary that maps a model name to an already-loaded `TextCrossEncoder` instance. This exists purely for performance: loading a reranker model involves reading ONNX weight files from disk, which is comparatively slow, so the cache ensures that cost is only paid once per model name rather than on every single rerank call.

### Lines 10-13 — `RerankError` exception
```python
class RerankError(Exception):
    """Raised when the reranker model can't be loaded, or fails to score
    the given candidates."""
```
- `class RerankError(Exception):` — a dedicated exception type for anything that goes wrong in this file, covering both "the model failed to load" and "the model failed to produce scores." Using one specific exception type lets calling code catch reranking failures distinctly from other kinds of errors elsewhere in the pipeline.

### Lines 15-26 — `_get_model`: lazily loading and caching the reranker
```python
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
```
- `def _get_model(model_name: str) -> TextCrossEncoder:` — a private helper (leading underscore signals it's internal to this module) that returns a loaded model instance for the given model name. Being private, it's not meant to be called from outside this file — `rerank()` below is the public entry point.
- The docstring explicitly says this mirrors the same caching rationale used in `sparse_client.py` elsewhere in the codebase, telling the reader this is a recognized, consistent pattern in the project rather than a one-off decision.
- `if model_name not in _model_cache:` — only does the expensive work of loading the model if it hasn't already been loaded and cached under this name.
- `try: _model_cache[model_name] = TextCrossEncoder(model_name=model_name)` — attempts to construct the cross-encoder model (which triggers loading its ONNX weights) and immediately stores it in the cache dictionary keyed by its name.
- `except Exception as exc: raise RerankError(...) from exc` — if construction fails for any reason (missing weights, corrupted download, unsupported model name, etc.), the raw exception is wrapped in the module's own `RerankError` with a message identifying which model failed to load. `from exc` preserves the original exception as the "cause" in the traceback, so debugging information isn't lost even though the exception type presented to callers is now the more specific `RerankError`.
- `return _model_cache[model_name]` — returns the cached (either just-created or previously cached) model instance.

### Lines 29-38 — `rerank` function signature and docstring
```python
def rerank(
    query: str, candidates: list[SearchCandidate], model_name: str, top_k: int
) -> list[SearchCandidate]:
    """Rerank `candidates` by relevance to `query` via a local cross-encoder,
    returning the best `top_k`.

    Each returned candidate's `score` is replaced with the cross-encoder's
    own relevance score - a more accurate signal than the fused hybrid
    search score for the chunks that actually reach generation.
    """
```
- `def rerank(query: str, candidates: list[SearchCandidate], model_name: str, top_k: int) -> list[SearchCandidate]:` — the public function of this module. It takes the original user query, the list of candidates found by hybrid search, which reranker model to use, and how many top results to keep, and returns a trimmed, re-scored list of the same `SearchCandidate` type.
- The docstring clarifies the key behavioral detail: the `score` field on each returned candidate is *overwritten* with the cross-encoder's score rather than kept as the original hybrid search score. This is called out explicitly because it's a meaningful design decision — the reasoning given is that the cross-encoder's score is a more trustworthy relevance signal specifically for the small set of chunks that will actually be shown to (or used by) the generation step, since it's the last and most accurate ranking pass before that happens.

### Lines 39-40 — Early return for no candidates
```python
    if not candidates:
        return []
```
- `if not candidates: return []` — guards against calling the model with an empty list. This avoids doing unnecessary work (loading a model, calling into it) when there's nothing to rerank, and avoids any edge-case errors the underlying library might raise on empty input.

### Lines 42-46 — Loading the model and scoring the candidates
```python
    model = _get_model(model_name)
    try:
        scores = list(model.rerank(query, [candidate.text for candidate in candidates]))
    except Exception as exc:
        raise RerankError(f"failed to rerank candidates: {exc}") from exc
```
- `model = _get_model(model_name)` — fetches the (possibly cached) cross-encoder instance for the requested model name.
- `scores = list(model.rerank(query, [candidate.text for candidate in candidates]))` — calls the cross-encoder's `rerank` method with the query and a plain list of each candidate's text content (extracted via a list comprehension). The model returns some iterable of scores, one per candidate text, in the same order they were passed in; `list(...)` materializes that iterable into a concrete list so it can be indexed/measured below.
- `except Exception as exc: raise RerankError(...) from exc` — wraps any failure during scoring (e.g. a malformed input, an internal model error) in the module's `RerankError`, again preserving the original exception via `from exc` for debugging, while presenting a single consistent exception type to callers.

### Lines 48-51 — Sanity-checking the score count
```python
    if len(scores) != len(candidates):
        raise RerankError(
            f"reranker returned {len(scores)} score(s) for {len(candidates)} candidate(s)"
        )
```
- `if len(scores) != len(candidates):` — defensively verifies that the model returned exactly one score per candidate that was submitted. This guards against a subtle and dangerous bug: if the counts mismatched, the `zip` used in the next step would silently pair up scores with the wrong candidates (or drop some), corrupting the ranking without any obvious error.
- `raise RerankError(...)` — if the counts don't match, the function fails loudly with a message stating exactly how many scores versus candidates there were, rather than silently proceeding with misaligned data.

### Lines 53-56 — Sorting by score and returning the top `top_k`
```python
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [
        replace(candidate, score=score) for score, candidate in ranked[:top_k]
    ]
```
- `ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)` — pairs up each score with its corresponding candidate using `zip` (relying on the fact that the score list and candidate list are in the same order, which was just verified above), then sorts those `(score, candidate)` pairs. `key=lambda pair: pair[0]` tells `sorted` to sort by the score (the first element of each pair), and `reverse=True` sorts from highest score to lowest, since higher cross-encoder scores mean more relevant.
- `return [replace(candidate, score=score) for score, candidate in ranked[:top_k]]` — takes only the top `top_k` entries from the sorted list (`ranked[:top_k]`), and for each one builds a new `SearchCandidate` using `dataclasses.replace`, copying all the original candidate's fields but substituting in the new cross-encoder `score`. This is necessary rather than just returning the candidates unchanged because `SearchCandidate` is a frozen dataclass (as defined in `search.py`), so its `score` field can't be mutated directly — `replace` is the idiomatic way to get an updated copy of an immutable object.
