from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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

    id: str
    query: str
    user_tier: str
    expected_answerable: bool
    expected_source_paths: list[str]


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
    raw = json.loads(path.read_text())
    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()

    for entry in raw:
        question_id = entry["id"]
        if question_id in seen_ids:
            raise ValueError(f"duplicate question id: {question_id!r}")
        seen_ids.add(question_id)

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
