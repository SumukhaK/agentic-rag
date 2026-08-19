# `evaluation/runner.py`

**Purpose:** This is the orchestrator for the entire structured evaluation process — it ties together everything else in `evaluation/` (the question fixtures, the faithfulness judge, the report builder) with the real production pipeline (retrieval, generation, ingestion) to actually run an evaluation end to end. It rebuilds a dedicated, isolated Qdrant (the vector database used for semantic search) collection from a known evaluation corpus, asks every evaluation question through the exact same code path a real user's query would take, scores each answer against its known-correct expectation, and produces both a JSON report file on disk and a structured log entry. The file is also a runnable script (`python -m agentic_rag.evaluation.runner`), so it can be executed directly to check the health of the whole system after a change.

## Line-by-line walkthrough

### Lines 1-26 — Imports
```python
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient

from agentic_rag.config import Settings
from agentic_rag.embedding.cache import EmbeddingCache
from agentic_rag.evaluation.judge import check_faithfulness
from agentic_rag.evaluation.questions import EvalQuestion, load_questions
from agentic_rag.evaluation.report import (
    EvalQuestionResult,
    EvaluationReport,
    build_report,
    report_to_json_dict,
)
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client
from agentic_rag.indexing.upsert import _point_id
from agentic_rag.ingestion.scheduler import run_sync_cycle
from agentic_rag.observability.eval_log import configure_eval_logging, log_evaluation_run
from agentic_rag.orchestration.answer import Citation
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE
from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache
```
- `from __future__ import annotations` — defers evaluation of type hints so modern syntax (like `bool | None`) works smoothly.
- `import json` — used to serialize the final report to a JSON file.
- `import time` — used for `time.monotonic()`, a clock that only ever moves forward and isn't affected by system clock adjustments, ideal for measuring durations.
- `from datetime import datetime` — used to timestamp the output report filename.
- `from pathlib import Path` — used for type hints and path construction (e.g. building the report file path).
- `from qdrant_client import QdrantClient` — the client class for talking to Qdrant, the vector database that stores document embeddings for semantic search; used here for type hints and to fetch cited source text.
- `from agentic_rag.config import Settings` — the app's central configuration object (environment variables, paths, model names, etc.).
- `from agentic_rag.embedding.cache import EmbeddingCache` — a cache that avoids recomputing the embedding (numeric vector representation) of identical text more than once.
- `from agentic_rag.evaluation.judge import check_faithfulness` — the faithfulness-judging function from `judge.py`, used to check whether a generated answer is backed up by its cited sources.
- `from agentic_rag.evaluation.questions import EvalQuestion, load_questions` — the question fixture type and loader from `questions.py`.
- `from agentic_rag.evaluation.report import (...)` — the result/report dataclasses and builder functions from `report.py`, imported together in a parenthesized multi-line import.
- `from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client` — helpers to obtain a Qdrant client and to make sure a named collection exists with the right configuration (e.g. vector dimensions).
- `from agentic_rag.indexing.upsert import _point_id` — the same deterministic ID-generation function the real indexing code uses to compute a Qdrant point's ID from a document's relative path and chunk index; imported here (even though it's named with a leading underscore, signaling it's "private" to its module) so this file can look up the same point another part of the system already wrote.
- `from agentic_rag.ingestion.scheduler import run_sync_cycle` — the same ingestion/indexing function used by the production background sync job, reused here so the evaluation corpus is indexed through the real code path rather than a separate eval-only reimplementation.
- `from agentic_rag.observability.eval_log import configure_eval_logging, log_evaluation_run` — logging helpers specific to evaluation runs, so an eval run's outcome lands in the same structured log stream as the rest of the system.
- `from agentic_rag.orchestration.answer import Citation` — the type representing one citation (a reference to a specific document chunk) attached to a generated answer.
- `from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE` — the exact canonical fallback text the system returns when it decides it can't answer a question, used here to detect whether a given answer was actually answered or just fell back.
- `from agentic_rag.orchestration.semantic_cache import SemanticCache, answer_with_cache` — the semantic caching layer and the top-level answer function; `answer_with_cache` is the same function the real `POST /query` API endpoint calls, so running evaluation questions through it means testing the real, complete pipeline.

### Lines 29-34 — `EvaluationIndexingError` exception class
```python
class EvaluationIndexingError(Exception):
    """Raised when indexing `eval/corpus/` itself fails - a corpus setup
    problem, not a pipeline measurement. Every question's score depends on
    the corpus having actually been indexed correctly; silently proceeding
    with a partially-indexed collection would produce a `retrieval_precision`
    indistinguishable from a real regression, for the wrong reason."""
```
- `class EvaluationIndexingError(Exception):` — a custom exception type, defined by simply subclassing the built-in `Exception` with only a docstring (no extra code needed). Its docstring makes clear that this exception is raised for a *setup* failure (something went wrong indexing the evaluation corpus itself), not a measurement of the actual pipeline being evaluated. This distinction matters because if the corpus wasn't indexed properly and the run proceeded anyway, a low `retrieval_precision` score would look identical to a real regression in the system, when actually the corpus was never correctly available to search in the first place — a misleading result.

### Lines 37-45 — `_normalize_path` helper
```python
def _normalize_path(relative_path: str) -> str:
    # Real relative_path values come from the ingestion pipeline's
    # str(path.relative_to(folder)), which uses os.sep - backslash on
    # Windows. expected_source_paths in the JSON fixture are written with
    # forward slashes for portability/readability. Normalizing both sides
    # to forward slashes before comparing avoids the exact "hardcoded
    # forward-slash literal vs. Windows backslash" test bug already hit
    # twice this session (test_scheduler.py, a live smoke-test script).
    return relative_path.replace("\\", "/")
```
- `def _normalize_path(relative_path: str) -> str:` — a small private helper function that takes a file path string and returns a normalized version.
- The comment explains the concrete bug this guards against: on Windows, the real ingestion pipeline produces paths using the OS's native separator (`\`), while the evaluation fixture file's expected paths are written with forward slashes (`/`) for portability and readability. Comparing these directly without normalizing would cause false mismatches purely due to slash direction, not any real difference in the paths — a bug the codebase had apparently already hit twice elsewhere.
- `return relative_path.replace("\\", "/")` — converts any backslashes in the given path to forward slashes, so paths from either source can be compared safely on an equal footing.

### Lines 48-78 — `_fetch_cited_source_text` function
```python
def _fetch_cited_source_text(
    client: QdrantClient, collection_name: str, citation: Citation
) -> str:
    """Fetch the actual indexed chunk text for `citation`, so the
    faithfulness judge can be shown what was really cited rather than
    just its `relative_path`/`chunk_index` metadata.
    ...
    Returns `""` if the point can't be found (a stale citation against a
    reindexed corpus) rather than raising - the faithfulness judge will
    then correctly see nothing to support the claim.
    """
    points = client.retrieve(
        collection_name=collection_name,
        ids=[_point_id(citation.relative_path, citation.chunk_index)],
        with_payload=True,
    )
    if not points:
        return ""
    return points[0].payload.get("text", "")
```
- `def _fetch_cited_source_text(client: QdrantClient, collection_name: str, citation: Citation) -> str:` — given a Qdrant client, the name of the collection to search, and a `Citation` (which only carries metadata — a document path and chunk index, not the actual text), this function fetches the real chunk text that citation refers to.
- The docstring explains why this lookup is needed at all: `Citation` deliberately doesn't carry the chunk's text because it's designed as lightweight metadata for the API response, not a full retrieval result. To let the faithfulness judge actually check the answer against real source text (not just a path/index reference), this function retrieves the stored payload — the same payload `index_document()` in `indexing/upsert.py` already wrote when the document was indexed. It looks the point up by its deterministic ID (computed with the same `_point_id()` function used at indexing time) via a direct `client.retrieve()` call by ID rather than a filtered `scroll()` search, because the ID is already known and re-deriving it a different way would be slower and risk drifting out of sync with the indexing code's own ID scheme.
- `points = client.retrieve(collection_name=collection_name, ids=[_point_id(citation.relative_path, citation.chunk_index)], with_payload=True)` — asks Qdrant to fetch the point (a stored vector plus its metadata/"payload") whose ID matches the deterministic ID computed from this citation's document path and chunk index; `with_payload=True` ensures the stored metadata (including the text) comes back, not just the ID.
- `if not points: return ""` — if no matching point was found (e.g. the citation refers to a chunk from an older version of the corpus that's since been reindexed and no longer exists under that ID), the function returns an empty string rather than raising an error.
- `return points[0].payload.get("text", "")` — otherwise, it pulls the `"text"` field out of the first (and only) matching point's stored payload dictionary, defaulting to an empty string if that key happens to be missing. This is the actual chunk text that will be shown to the faithfulness judge as evidence.

### Lines 81-145 — `_run_question` signature and docstring
```python
def _run_question(
    question: EvalQuestion,
    *,
    settings: Settings,
    client: QdrantClient,
    embedding_cache: EmbeddingCache,
) -> EvalQuestionResult:
    """Answer `question` through the real pipeline (`answer_with_cache()`
    ...
    have to be added to two near-identical constructor calls in lockstep.
    """
```
- `def _run_question(question: EvalQuestion, *, settings: Settings, client: QdrantClient, embedding_cache: EmbeddingCache) -> EvalQuestionResult:` — this is the function that answers a single evaluation question through the real pipeline and scores the result; `*` forces `settings`, `client`, and `embedding_cache` to be passed by keyword only.
- The docstring documents several important design choices, summarized here: (1) a *fresh* `SemanticCache` is created for each individual question, not shared across the whole run, because `SemanticCache` serves cached results for *similar-enough* (not just identical) queries — sharing one cache across the whole run risks a later question being silently scored against an unrelated earlier question's cached answer if their embeddings happen to be close, a real risk as the question set grows to include paraphrased or near-duplicate questions. `embedding_cache`, by contrast, is safe to share across questions because it's keyed by *exact* text rather than fuzzy similarity, so sharing it only ever saves redundant computation without any risk of cross-contamination. (2) `retrieval_hit` is computed only for questions expected to be answerable, staying `None` otherwise so it doesn't distort the aggregate metric. (3) Faithfulness is judged only when the system actually produced an answer to a question that was expected to be answerable — skipping the judge call in other cases either because there's nothing to check (fallback given) or because the outcome is already fully determined without an extra model call. (4) `hallucinated` has two distinct triggers: answering at all when the question shouldn't have been answerable (fabrication), or answering unfaithfully to sourced material when it should have been answerable (an unsupported claim) — a question that should have been answerable but got the fallback instead is neither of these; it's a retrieval miss, a separate failure mode. (5) Any exception is caught and recorded in the result's `error` field instead of propagating, so one question's infrastructure failure (e.g. a judge-model timeout) doesn't wipe out every other already-computed result in the run — mirroring the same per-item isolation `run_sync_cycle()` applies during indexing. (6) `duration_seconds` is measured with `time.monotonic()` (immune to wall-clock adjustments) from entry to the single return statement, covering the whole round trip including failures.

### Lines 146-158 — Starting the timer and calling the real pipeline
```python
    start = time.monotonic()
    try:
        cache = SemanticCache()
        answer = answer_with_cache(
            question.query,
            question.user_tier,
            cache=cache,
            client=client,
            collection_name=settings.evaluation_qdrant_collection_name,
            embedding_cache=embedding_cache,
            known_tiers=settings.access_tiers,
            settings=settings,
        )
```
- `start = time.monotonic()` — records the current value of the monotonic clock as the starting point for this question's timing measurement.
- `try:` — opens a block that wraps the whole answer-and-score process, so any exception raised inside can be caught below rather than crashing the whole evaluation run.
- `cache = SemanticCache()` — creates a brand-new, empty semantic cache dedicated to this one question, per the reasoning in the docstring above.
- `answer = answer_with_cache(question.query, question.user_tier, cache=cache, client=client, collection_name=settings.evaluation_qdrant_collection_name, embedding_cache=embedding_cache, known_tiers=settings.access_tiers, settings=settings)` — calls the actual production answer-generation function, the same one the live `POST /query` endpoint uses, passing: the question's text and access tier, the fresh per-question cache, the shared Qdrant client, the dedicated evaluation collection name (not the production one), the shared embedding cache, the set of known access tiers, and the app settings. This is the core of what makes this an evaluation of the *real* system rather than a mock.

### Lines 160-167 — Determining whether the system actually answered
```python
        # Substring containment, not exact equality - matches
        # answer_with_cache()'s own convention (`CANNOT_ANSWER_MESSAGE not
        # in answer.text`). Live-verified real gap: the generation model
        # sometimes wraps the fallback in surrounding whitespace (e.g. a
        # leading space), which `!=` fails to recognize as the fallback
        # at all, miscounting a correct "I don't know" as a fabricated
        # answer.
        answered = CANNOT_ANSWER_MESSAGE not in answer.text
```
- The comment explains a subtle correctness detail: checking whether the returned answer text equals the fallback message exactly (`!=`) would fail in practice, because the generation model sometimes returns the fallback message wrapped in extra whitespace, causing an exact-equality check to wrongly treat a correct "I don't know" response as if it were a fabricated answer.
- `answered = CANNOT_ANSWER_MESSAGE not in answer.text` — instead, this checks whether the canonical fallback text appears *anywhere within* the returned text (substring containment), matching the same convention `answer_with_cache()` itself already uses internally. If the fallback message isn't found anywhere in the text, the system is considered to have "answered" for real.

### Lines 168-173 — Normalizing citations and computing `retrieval_hit`
```python
        cited_paths = [_normalize_path(c.relative_path) for c in answer.citations]

        retrieval_hit = None
        if question.expected_answerable:
            expected = {_normalize_path(p) for p in question.expected_source_paths}
            retrieval_hit = bool(expected & set(cited_paths))
```
- `cited_paths = [_normalize_path(c.relative_path) for c in answer.citations]` — builds a list of the document paths actually cited in the answer, running each through `_normalize_path()` so slash direction doesn't cause false mismatches.
- `retrieval_hit = None` — starts out as `None`, meaning "this metric doesn't apply," and stays that way unless the block below sets it.
- `if question.expected_answerable:` — retrieval precision is only meaningful for a question that's actually expected to be answerable from the corpus.
- `expected = {_normalize_path(p) for p in question.expected_source_paths}` — builds a *set* (for fast membership/intersection checks) of the expected ground-truth source paths, also normalized.
- `retrieval_hit = bool(expected & set(cited_paths))` — computes the intersection (`&`) between the expected paths and the actually-cited paths; if there's any overlap at all, `retrieval_hit` becomes `True` (wrapped with `bool()` to convert the resulting set into an actual boolean — a non-empty set is truthy, an empty one is falsy).

### Lines 175-190 — Running the faithfulness check when applicable
```python
        faithfulness = None
        if answered and question.expected_answerable:
            sources = "\n\n".join(
                f"[{citation.number}] "
                f"{_fetch_cited_source_text(client, settings.evaluation_qdrant_collection_name, citation)}"
                for citation in answer.citations
            )
            faithfulness = check_faithfulness(
                question.query,
                answer.text,
                sources,
                model=settings.evaluation_model,
                base_url=settings.ollama_base_url,
                timeout=settings.evaluation_timeout_seconds,
                temperature=settings.evaluation_temperature,
            )
```
- `faithfulness = None` — defaults to "not judged," matching the same "doesn't apply" convention used for `retrieval_hit`.
- `if answered and question.expected_answerable:` — faithfulness is only checked if the system produced a real answer *and* the question was actually expected to be answerable — per the docstring's reasoning, checking it in other cases would either have nothing to check against or would just re-confirm an already-known outcome.
- `sources = "\n\n".join(f"[{citation.number}] {_fetch_cited_source_text(...)}" for citation in answer.citations)` — builds a single string containing the text of every cited source, each one prefixed with its citation number in brackets (matching how citations are likely numbered in the answer itself) and separated by blank lines, using the `_fetch_cited_source_text()` helper defined above to pull the real chunk text for each citation.
- `faithfulness = check_faithfulness(question.query, answer.text, sources, model=settings.evaluation_model, base_url=settings.ollama_base_url, timeout=settings.evaluation_timeout_seconds, temperature=settings.evaluation_temperature)` — calls the faithfulness judge from `judge.py`, passing the original question, the generated answer text, the assembled source text, and the evaluation-specific model/connection/timing settings (kept separate from whatever settings the generation model itself uses).

### Lines 192-197 — Computing `hallucinated`
```python
        if not question.expected_answerable:
            hallucinated = answered
        else:
            hallucinated = (
                answered and faithfulness is not None and not faithfulness.is_faithful
            )
```
- `if not question.expected_answerable: hallucinated = answered` — for a question that shouldn't have been answerable at all, simply answering it (regardless of what was said) already counts as hallucination — the system fabricated an answer where the corpus had nothing to support one.
- `else: hallucinated = (answered and faithfulness is not None and not faithfulness.is_faithful)` — for a question that *should* have been answerable, hallucination instead means: the system did answer, a faithfulness verdict was actually computed (`faithfulness is not None`), and that verdict says the answer was *not* faithful to its cited sources. Note that if the system gave the fallback message instead of answering (a retrieval miss, not fabrication), `answered` is `False` and this whole expression is `False` — that case is deliberately not counted as a hallucination.

### Lines 199-207 — Building the success outcome dictionary
```python
        outcome = dict(
            answer_text=answer.text,
            cited_paths=cited_paths,
            retrieval_hit=retrieval_hit,
            answered=answered,
            faithfulness=faithfulness,
            hallucinated=hallucinated,
            error=None,
        )
```
- `outcome = dict(...)` — collects all the fields computed above into a plain dictionary named `outcome`, representing the successful-path result. Using `dict(...)` with keyword arguments here (rather than building the final `EvalQuestionResult` directly) lets this block and the exception-handling block below both produce a same-shaped dictionary that gets unpacked into the final constructor call later, so the fields that both paths share don't need to be repeated in two different `EvalQuestionResult(...)` calls.
- `error=None` — explicitly marks this as a non-errored outcome.

### Lines 208-217 — Catching and isolating any failure
```python
    except Exception as exc:  # noqa: BLE001 - isolate one bad question, see docstring
        outcome = dict(
            answer_text="",
            cited_paths=[],
            retrieval_hit=None,
            answered=False,
            faithfulness=None,
            hallucinated=False,
            error=f"{type(exc).__name__}: {exc}",
        )
```
- `except Exception as exc:` — catches literally any exception raised anywhere in the `try` block above (a broad catch, which the `# noqa: BLE001` comment acknowledges and justifies as intentional — `BLE001` is a linter rule that normally flags catching a bare/broad `Exception` as too permissive, but it's deliberately allowed here so one bad question can't take down the whole evaluation run).
- `outcome = dict(answer_text="", cited_paths=[], retrieval_hit=None, answered=False, faithfulness=None, hallucinated=False, error=f"{type(exc).__name__}: {exc}")` — builds a placeholder outcome where every field is a "nothing happened" default value, and records the exception's class name and message as a human-readable string in `error`. As `report.py`'s docstrings emphasize, these placeholder values (like `hallucinated=False`) are not real measurements and get excluded from the correctness metrics precisely because `error` is set here.

### Lines 219-226 — Building and returning the final result
```python
    return EvalQuestionResult(
        question_id=question.id,
        query=question.query,
        expected_answerable=question.expected_answerable,
        expected_source_paths=question.expected_source_paths,
        duration_seconds=time.monotonic() - start,
        **outcome,
    )
```
- `return EvalQuestionResult(question_id=question.id, query=question.query, expected_answerable=question.expected_answerable, expected_source_paths=question.expected_source_paths, duration_seconds=time.monotonic() - start, **outcome)` — constructs the final result object. The fields that come directly from the original `question` (its ID, text, and expectations) are supplied explicitly here — the one place they're needed regardless of success or failure. `duration_seconds=time.monotonic() - start` computes the elapsed time by subtracting the starting timestamp from the current one, covering both success and failure paths since this line runs either way. `**outcome` unpacks whichever outcome dictionary was built above (success or failure) as the remaining keyword arguments, filling in `answer_text`, `cited_paths`, `retrieval_hit`, `answered`, `faithfulness`, `hallucinated`, and `error`.

### Lines 229-275 — `run_evaluation` signature and docstring
```python
def run_evaluation(*, settings: Settings) -> EvaluationReport:
    """Run the full structured evaluation: index `settings.evaluation_corpus_path`
    ...
    the same reasoning `api/app.py`'s `lifespan` already documents for
    the production client.
    """
```
- `def run_evaluation(*, settings: Settings) -> EvaluationReport:` — the top-level function that runs an entire evaluation from start to finish: index the evaluation corpus, answer every question, and return the aggregate report. `*` forces `settings` to be passed by keyword.
- The docstring lays out several important design decisions: (1) a completely separate Qdrant storage path and collection name are used for evaluation, so an eval run can never accidentally mix fixture-only documents into production data, or read against real production data during a test. (2) The evaluation collection is deleted and rebuilt from scratch at the start of *every* run rather than being reused. This matters because `run_sync_cycle()` (the production ingestion function reused here) is designed for a long-running background loop that persists its "snapshot" of what's already indexed between cycles — but this function always starts from an empty snapshot (`previous_snapshot={}`). Against a *persisted* collection, that would mean a document removed from the corpus between eval runs would never get cleaned out of the index (the diffing logic can't detect a deletion if it doesn't know what used to be there). Deleting and recreating the collection first makes the empty-snapshot assumption actually correct rather than just convenient — every run starts genuinely empty. (3) The corpus is indexed via the real `run_sync_cycle()` function — the exact code the production background sync job uses — not a separate eval-only reimplementation, ensuring the evaluation measures the real system's actual behavior. Its result is checked, and any failure raises `EvaluationIndexingError` rather than silently continuing with an incompletely indexed collection. (4) A modified copy of `settings` (`eval_settings`) is used only for the indexing call itself, since only that call needs the corpus path and collection name overridden — `_run_question()` reads the evaluation collection name straight from the original `settings` object it already receives, so there's no need to pass the modified copy any further. (5) The Qdrant client is always closed in a `finally` block, because an embedded/local-mode Qdrant instance holds an exclusive file lock on its storage directory for as long as the client stays open — the same reasoning already documented for the production client's lifecycle elsewhere in the codebase.

### Line 276 — Loading the questions
```python
    questions = load_questions(settings.evaluation_questions_path)
```
- `questions = load_questions(settings.evaluation_questions_path)` — loads and validates the full evaluation question set from the path configured in settings, using the `load_questions()` function from `questions.py`.

### Lines 278-284 — Opening the Qdrant client and rebuilding the collection
```python
    client = get_client(settings.evaluation_qdrant_storage_path)
    try:
        if client.collection_exists(settings.evaluation_qdrant_collection_name):
            client.delete_collection(settings.evaluation_qdrant_collection_name)
        ensure_collection(
            client, settings.evaluation_qdrant_collection_name, settings.embedding_dimensions
        )
```
- `client = get_client(settings.evaluation_qdrant_storage_path)` — opens a Qdrant client pointed at the dedicated evaluation storage path (not the production one), using the shared `get_client()` helper.
- `try:` — opens a block whose `finally` (further below) guarantees the client gets closed no matter what happens inside.
- `if client.collection_exists(settings.evaluation_qdrant_collection_name): client.delete_collection(settings.evaluation_qdrant_collection_name)` — if a collection from a previous evaluation run already exists under this name, it's deleted first, so this run starts from a guaranteed-clean slate (per the docstring's reasoning above).
- `ensure_collection(client, settings.evaluation_qdrant_collection_name, settings.embedding_dimensions)` — (re)creates the collection fresh, configured with the correct vector dimensionality for the embeddings that will be stored in it.

### Lines 286-294 — Indexing the evaluation corpus
```python
        eval_settings = settings.model_copy(
            update={
                "watched_folder_path": settings.evaluation_corpus_path,
                "qdrant_collection_name": settings.evaluation_qdrant_collection_name,
            }
        )
        sync_result, _snapshot = run_sync_cycle(
            settings=eval_settings, client=client, previous_snapshot={}
        )
```
- `eval_settings = settings.model_copy(update={"watched_folder_path": settings.evaluation_corpus_path, "qdrant_collection_name": settings.evaluation_qdrant_collection_name})` — creates a copy of the app settings (using Pydantic's `model_copy()`, since `Settings` is presumably a Pydantic model) with two fields overridden: the folder to watch/index becomes the evaluation corpus's folder, and the target collection becomes the evaluation collection, instead of whatever the production settings normally point to.
- `sync_result, _snapshot = run_sync_cycle(settings=eval_settings, client=client, previous_snapshot={})` — runs the real production ingestion/indexing cycle against the evaluation corpus, using the modified settings and starting from an empty `previous_snapshot` (so everything in the corpus looks "new" and gets indexed). It returns both a `sync_result` (details about what succeeded/failed) and a new snapshot, which is discarded here (assigned to `_snapshot`, the underscore-prefixed name signaling it's intentionally unused) since this function doesn't need to persist state between evaluation runs.

### Lines 295-306 — Checking for indexing failures
```python
        if (
            sync_result.ingestion_failures
            or sync_result.indexing_failures
            or sync_result.deletion_failures
        ):
            raise EvaluationIndexingError(
                "failed to index the eval corpus: "
                f"{len(sync_result.ingestion_failures)} ingestion failure(s), "
                f"{len(sync_result.indexing_failures)} indexing failure(s), "
                f"{len(sync_result.deletion_failures)} deletion failure(s) - "
                f"{[f.relative_path for f in sync_result.ingestion_failures]}"
            )
```
- `if (sync_result.ingestion_failures or sync_result.indexing_failures or sync_result.deletion_failures):` — checks whether any of the three failure lists on the sync result are non-empty (a non-empty list is truthy in Python), i.e. whether anything went wrong while reading, indexing, or deleting documents during corpus setup.
- `raise EvaluationIndexingError(...)` — if any failures occurred, the function stops immediately with a detailed error message rather than silently proceeding to score every question against a partially-indexed collection. The message includes counts of each failure type and the specific relative paths of the documents that failed to ingest, giving a developer enough detail to diagnose the problem without needing to dig through logs.

### Lines 308-317 — Answering every question
```python
        embedding_cache = EmbeddingCache()
        results = [
            _run_question(
                question,
                settings=settings,
                client=client,
                embedding_cache=embedding_cache,
            )
            for question in questions
        ]
```
- `embedding_cache = EmbeddingCache()` — creates one shared embedding cache for the whole run, reused across every question since (as explained earlier) it's safe to share — it only avoids recomputing identical embeddings, with no risk of cross-question contamination.
- `results = [_run_question(question, settings=settings, client=client, embedding_cache=embedding_cache) for question in questions]` — a list comprehension that calls `_run_question()` once for every question in the loaded question set, passing the *original* (unmodified) `settings`, the shared Qdrant client, and the shared embedding cache. Note this passes `settings`, not `eval_settings` — `_run_question()` reads the evaluation collection name directly off `settings` itself.

### Lines 318-321 — Closing the client and returning the report
```python
    finally:
        client.close()

    return build_report(results)
```
- `finally: client.close()` — regardless of whether the code above succeeded or raised an exception, this guarantees the Qdrant client is always closed, releasing its exclusive file lock on the storage directory.
- `return build_report(results)` — passes the full list of per-question results to `build_report()` (from `report.py`), which computes the aggregate metrics, and returns the resulting `EvaluationReport` to the caller.

### Lines 324-326 — `_report_path_for` helper
```python
def _report_path_for(results_dir: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return results_dir / f"eval-{timestamp}.json"
```
- `def _report_path_for(results_dir: Path, now: datetime | None = None) -> Path:` — a small helper that computes the output file path for a report; accepting an optional `now` parameter (defaulting to `None`) makes the current time injectable, which is a common pattern for keeping code testable without relying on the real system clock.
- `timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")` — uses the passed-in `now` if one was given, otherwise falls back to the real current time (`datetime.now()`), and formats it as a compact timestamp string like `20260819T143000`.
- `return results_dir / f"eval-{timestamp}.json"` — builds the final path by joining the results directory with a filename like `eval-20260819T143000.json`, using `Path`'s `/` operator for path joining. Because each run gets its own uniquely-timestamped filename, repeated evaluation runs accumulate a history of reports instead of overwriting each other.

### Lines 329-343 — `main` function docstring
```python
def main() -> None:
    """CLI entry point: `python -m agentic_rag.evaluation.runner`.
    ...
    JSON-lines stream every other component (`POST /query`, the
    background sync job) logs through, not just this script's own stdout.
    """
```
- `def main() -> None:` — the function that runs when this module is executed as a script.
- The docstring explains this is the command-line entry point (`python -m agentic_rag.evaluation.runner`), and documents its behavior: it loads `Settings` from the environment/`.env` file exactly the same way the FastAPI app does, so an evaluation run always measures the system as it's actually configured, not some separately-configured shadow version; it writes the full report as timestamped JSON so repeated runs build up a history; it prints the summary metrics to the console for interactive use; and it also emits one structured JSON log line via `configure_eval_logging()`/`log_evaluation_run()`, so the evaluation run's outcome ends up in the same log stream used by every other part of the system (the query endpoint, the background sync job), not just this script's own console output.

### Lines 344-348 — Setting up logging and running the evaluation
```python
    configure_eval_logging()
    run_start = time.monotonic()

    settings = Settings()
    report = run_evaluation(settings=settings)
```
- `configure_eval_logging()` — sets up the logging configuration specific to evaluation runs (defined in `observability/eval_log.py`), so the subsequent `log_evaluation_run()` call actually produces output in the right format/location.
- `run_start = time.monotonic()` — records the start time of the entire run (as opposed to `_run_question`'s per-question `start`), used later to compute the total run duration.
- `settings = Settings()` — instantiates the app's settings object, which (per its `pydantic-settings` design, per the project's architecture) reads configuration from environment variables and the `.env` file.
- `report = run_evaluation(settings=settings)` — runs the entire evaluation process described above and captures the resulting `EvaluationReport`.

### Lines 350-352 — Writing the report to disk
```python
    settings.evaluation_results_path.mkdir(parents=True, exist_ok=True)
    report_path = _report_path_for(settings.evaluation_results_path)
    report_path.write_text(json.dumps(report_to_json_dict(report), indent=2))
```
- `settings.evaluation_results_path.mkdir(parents=True, exist_ok=True)` — ensures the directory where reports are stored exists, creating any missing parent directories (`parents=True`) and not raising an error if it already exists (`exist_ok=True`).
- `report_path = _report_path_for(settings.evaluation_results_path)` — computes the timestamped output file path using the helper defined above.
- `report_path.write_text(json.dumps(report_to_json_dict(report), indent=2))` — converts the report to a plain JSON-serializable dictionary (via `report_to_json_dict()` from `report.py`), serializes it to a pretty-printed JSON string (`indent=2` for human readability), and writes it to the computed file path.

### Lines 354-361 — Logging the structured summary
```python
    log_evaluation_run(
        retrieval_precision=report.retrieval_precision,
        faithfulness_rate=report.faithfulness_rate,
        hallucination_rate=report.hallucination_rate,
        errored_count=report.errored_count,
        average_duration_seconds=report.average_duration_seconds,
        report_path=str(report_path),
        run_duration_seconds=time.monotonic() - run_start,
    )
```
- `log_evaluation_run(retrieval_precision=..., faithfulness_rate=..., hallucination_rate=..., errored_count=..., average_duration_seconds=..., report_path=str(report_path), run_duration_seconds=time.monotonic() - run_start)` — calls the structured logging helper with all the summary metrics from the report, the path where the full report was written (converted to a string since `report_path` is a `Path` object), and the total time the entire evaluation run took to execute (current monotonic time minus `run_start`).

### Lines 364-369 — Printing the summary to the console
```python
    print(f"retrieval_precision: {report.retrieval_precision}")
    print(f"faithfulness_rate:   {report.faithfulness_rate}")
    print(f"hallucination_rate:  {report.hallucination_rate}")
    print(f"errored_count:       {report.errored_count}")
    print(f"avg_duration_seconds:{report.average_duration_seconds}")
    print(f"report written to:   {report_path}")
```
- Each `print(...)` line writes one summary line to standard output, with labels aligned via manual spacing so a developer running this script interactively gets an immediately readable summary: retrieval precision, faithfulness rate, hallucination rate, the number of errored questions, the average per-question duration, and finally the path of the full JSON report file that was written, in case they want to inspect it further.

### Lines 372-373 — Script entry point guard
```python
if __name__ == "__main__":
    main()
```
- `if __name__ == "__main__":` — the standard Python idiom that checks whether this file is being run directly as a script (e.g. via `python -m agentic_rag.evaluation.runner`) rather than being imported as a module by some other code.
- `main()` — if run directly, calls the `main()` function defined above, kicking off the entire evaluation process, report writing, and logging.
