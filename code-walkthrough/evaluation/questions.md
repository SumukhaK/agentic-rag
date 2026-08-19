# `evaluation/questions.py`

**Purpose:** This file defines the shape of a single evaluation question (a hand-written test case with a known-correct expected outcome) and the function that loads a whole set of these questions from a JSON file on disk. These questions are the fixed "exam" the evaluation runner (`runner.py`) uses to check whether the real RAG (retrieval-augmented generation) pipeline is behaving correctly — retrieving the right documents, answering when it should, and refusing to answer (rather than making something up) when it shouldn't. Because a badly-formed question could silently corrupt the metrics computed from it, this file validates each question strictly and raises loud errors instead of quietly accepting bad data.

## Line-by-line walkthrough

### Lines 1-6 — Imports
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
```
- `from __future__ import annotations` — makes all type hints in this file lazily evaluated as strings rather than actual objects at import time. This lets, e.g., `list[str]` be used as a type hint even in Python versions where that syntax normally needs extra care, and generally speeds up import.
- `import json` — the standard library module used to parse the JSON file the question set is stored in.
- `from dataclasses import dataclass` — the decorator used to build the plain, immutable `EvalQuestion` record below.
- `from pathlib import Path` — used to type the `path` parameter of `load_questions()` and to call `.read_text()` on it.

### Lines 9-19 — `EvalQuestion` dataclass and its docstring
```python
@dataclass(frozen=True)
class EvalQuestion:
    """One hand-curated evaluation fixture: a question with a known-correct
    expectation to check the real pipeline's output against.

    `expected_answerable=False` marks a question deliberately unanswerable
    from the eval corpus - used to measure hallucination rate (does the
    system correctly fall back instead of fabricating an answer).
    `expected_source_paths` is the ground truth `retrieval_precision()`
    checks actual citations against; it's meaningless (and required to be
    empty) when the question isn't expected to be answerable at all.
    """
```
- `@dataclass(frozen=True)` — again marks this as an immutable value object; a loaded `EvalQuestion` can't have its fields changed after construction.
- `class EvalQuestion:` — represents one row/entry of the evaluation fixture file — a single test case.
- The docstring explains the intent behind the two trickiest fields: `expected_answerable=False` deliberately marks a question that the system should *not* be able to answer from the evaluation corpus (used to test whether the system correctly says "I don't know" instead of hallucinating an answer), and `expected_source_paths` is the "ground truth" (the objectively correct answer to compare against) used to check whether the system cited the right documents — it only makes sense, and is required to be empty, when the question is not expected to be answerable at all.

### Lines 21-25 — `EvalQuestion` fields
```python
    id: str
    query: str
    user_tier: str
    expected_answerable: bool
    expected_source_paths: list[str]
```
- `id: str` — a unique identifier for this question, used to detect duplicates and to reference a specific question in a report.
- `query: str` — the actual natural-language question text sent through the pipeline.
- `user_tier: str` — the access tier (permission level) the question should be asked as, since the system may restrict which documents a given tier can see.
- `expected_answerable: bool` — whether the evaluation corpus actually contains enough information to answer this question correctly.
- `expected_source_paths: list[str]` — the list of document paths that should be cited if the system answers this question correctly; this is the "ground truth" used to score retrieval precision (how much of what was retrieved was actually relevant).

### Lines 28-47 — `load_questions` function signature and docstring
```python
def load_questions(path: Path) -> list[EvalQuestion]:
    """Load the evaluation question set from `path` (a JSON array).

    Plain JSON, not YAML - the question set is a flat list of short,
    single-line records with no nested structure or multi-line values, so
    YAML's main advantage (human-editable long/nested config) doesn't
    apply, and stdlib `json` avoids adding a dependency for it.

    Raises `ValueError` loudly for a malformed fixture rather than
    silently producing a question that would skip its own metric, or
    carry stale/contradictory data into the report: a duplicate `id`
    (ambiguous which fixture a report line refers to),
    `expected_answerable=True` with no `expected_source_paths` (would
    silently exclude itself from the retrieval-precision calculation
    instead of ever counting as a miss), or `expected_answerable=False`
    with a non-empty `expected_source_paths` (the reverse contradiction -
    most likely a fixture copied from an answerable question with
    `expected_answerable` flipped but its now-meaningless expected
    sources left behind).
    """
```
- `def load_questions(path: Path) -> list[EvalQuestion]:` — takes a filesystem path to a JSON file and returns the parsed, validated list of question objects.
- The docstring explains two design decisions: (1) plain JSON was chosen over YAML because the data is a flat list of short records with no nested or multi-line values, so YAML's main benefit (readability for complex nested/long config) doesn't apply, and using Python's built-in `json` module avoids adding a third-party dependency; (2) the function raises `ValueError` loudly on any malformed entry — a duplicate `id`, an answerable question missing its expected sources, or an unanswerable question that still has expected sources — because silently accepting bad data would either make a report line ambiguous (which fixture does it refer to?) or silently corrupt a metric (e.g. a question that should count as a retrieval miss instead just gets skipped from the calculation entirely).

### Lines 48-50 — Reading and parsing the file
```python
    raw = json.loads(path.read_text())
    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
```
- `raw = json.loads(path.read_text())` — reads the entire file at `path` as text and parses it as JSON, producing a Python list of dictionaries (one per question entry).
- `questions: list[EvalQuestion] = []` — an empty list that will accumulate the validated `EvalQuestion` objects as they're built.
- `seen_ids: set[str] = set()` — an empty set used to track which `id` values have already been seen, so duplicates can be detected as the loop below processes each entry.

### Lines 52-56 — Looping and checking for duplicate IDs
```python
    for entry in raw:
        question_id = entry["id"]
        if question_id in seen_ids:
            raise ValueError(f"duplicate question id: {question_id!r}")
        seen_ids.add(question_id)
```
- `for entry in raw:` — iterates over each raw dictionary parsed from the JSON array.
- `question_id = entry["id"]` — pulls the `id` field out of the entry (using `[...]`, not `.get(...)`, so a missing `id` key raises a `KeyError` immediately rather than being silently treated as absent).
- `if question_id in seen_ids: raise ValueError(...)` — if this ID was already processed earlier in the loop, the function stops immediately with a clear error message rather than allowing two questions to share an ID (which would make later report entries ambiguous about which fixture they refer to).
- `seen_ids.add(question_id)` — records this ID as seen, so a later duplicate can be caught.

### Lines 58-70 — Validating `expected_answerable` vs `expected_source_paths` consistency
```python
        expected_answerable = entry["expected_answerable"]
        expected_source_paths = entry.get("expected_source_paths", [])
        if expected_answerable and not expected_source_paths:
            raise ValueError(
                f"question {question_id!r} is expected_answerable but has no "
                "expected_source_paths to check retrieval precision against"
            )
        if not expected_answerable and expected_source_paths:
            raise ValueError(
                f"question {question_id!r} is not expected_answerable but has "
                f"non-empty expected_source_paths ({expected_source_paths!r}) - "
                "these must be empty for a question with nothing to retrieve"
            )
```
- `expected_answerable = entry["expected_answerable"]` — required field, pulled directly (missing it raises `KeyError`).
- `expected_source_paths = entry.get("expected_source_paths", [])` — optional field; if absent, defaults to an empty list rather than erroring, since it's legitimately empty for unanswerable questions.
- `if expected_answerable and not expected_source_paths: raise ValueError(...)` — catches the first kind of contradiction: a question marked as answerable but with no expected source documents listed. Without this check, such a question would silently be excluded from the retrieval-precision metric instead of ever being counted as a miss, which would make the metric look better than it should.
- `if not expected_answerable and expected_source_paths: raise ValueError(...)` — catches the opposite contradiction: a question marked as unanswerable but which still lists expected source documents. The docstring notes this is most likely a leftover from copying an answerable question's fixture and flipping the flag without clearing the now-meaningless source list.

### Lines 72-82 — Building the `EvalQuestion` and returning the full list
```python
        questions.append(
            EvalQuestion(
                id=question_id,
                query=entry["query"],
                user_tier=entry["user_tier"],
                expected_answerable=expected_answerable,
                expected_source_paths=expected_source_paths,
            )
        )

    return questions
```
- `questions.append(EvalQuestion(...))` — once an entry passes all validation, it's turned into an actual `EvalQuestion` dataclass instance and appended to the results list.
- `id=question_id, query=entry["query"], user_tier=entry["user_tier"], expected_answerable=expected_answerable, expected_source_paths=expected_source_paths` — maps each raw JSON field to the corresponding dataclass field; `query` and `user_tier` are required and accessed directly with `[...]` (missing either raises `KeyError`), while `expected_answerable`/`expected_source_paths` reuse the already-validated local variables from above.
- `return questions` — after the loop finishes processing every entry in the file, the function returns the complete, validated list of `EvalQuestion` objects to the caller.
