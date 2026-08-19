import json

import pytest

from agentic_rag.evaluation.questions import EvalQuestion, load_questions
from access_tiers import TIER_EMPLOYEE


def _write(path, payload):
    path.write_text(json.dumps(payload))


def test_load_questions_parses_an_answerable_question(tmp_path):
    path = tmp_path / "questions.json"
    _write(
        path,
        [
            {
                "id": "q1",
                "query": "Who won the derby?",
                "user_tier": TIER_EMPLOYEE,
                "expected_answerable": True,
                "expected_source_paths": ["employee/derby.md"],
            }
        ],
    )

    questions = load_questions(path)

    assert questions == [
        EvalQuestion(
            id="q1",
            query="Who won the derby?",
            user_tier=TIER_EMPLOYEE,
            expected_answerable=True,
            expected_source_paths=["employee/derby.md"],
        )
    ]


def test_load_questions_defaults_expected_source_paths_to_empty(tmp_path):
    # An unanswerable question has no expected source - requiring every
    # entry to spell out an empty list would be pure boilerplate.
    path = tmp_path / "questions.json"
    _write(
        path,
        [
            {
                "id": "q2",
                "query": "Who won the 1850 derby?",
                "user_tier": TIER_EMPLOYEE,
                "expected_answerable": False,
            }
        ],
    )

    questions = load_questions(path)

    assert questions[0].expected_source_paths == []


def test_load_questions_preserves_file_order(tmp_path):
    path = tmp_path / "questions.json"
    _write(
        path,
        [
            {"id": "a", "query": "?", "user_tier": TIER_EMPLOYEE, "expected_answerable": True,
             "expected_source_paths": ["x"]},
            {"id": "b", "query": "?", "user_tier": TIER_EMPLOYEE, "expected_answerable": True,
             "expected_source_paths": ["y"]},
        ],
    )

    questions = load_questions(path)

    assert [q.id for q in questions] == ["a", "b"]


def test_load_questions_rejects_a_duplicate_id(tmp_path):
    path = tmp_path / "questions.json"
    _write(
        path,
        [
            {"id": "a", "query": "?", "user_tier": TIER_EMPLOYEE, "expected_answerable": True,
             "expected_source_paths": ["x"]},
            {"id": "a", "query": "??", "user_tier": TIER_EMPLOYEE, "expected_answerable": True,
             "expected_source_paths": ["y"]},
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_load_questions_rejects_an_answerable_question_with_no_expected_sources(tmp_path):
    # A question marked answerable but with nothing to check retrieval
    # precision against is a malformed fixture, not a valid "unanswerable"
    # question - it would silently skip the retrieval-precision metric
    # instead of ever failing loudly.
    path = tmp_path / "questions.json"
    _write(
        path,
        [
            {
                "id": "q1",
                "query": "Who won?",
                "user_tier": TIER_EMPLOYEE,
                "expected_answerable": True,
                "expected_source_paths": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="expected_source_paths"):
        load_questions(path)


def test_load_questions_rejects_an_unanswerable_question_with_expected_sources(tmp_path):
    # The reverse malformed fixture: expected_answerable=False is
    # documented as requiring an empty expected_source_paths (there's
    # nothing to check retrieval precision against for a question that
    # isn't supposed to be answerable at all) - a non-empty list here is
    # stale/contradictory data, most likely from copying an answerable
    # question and forgetting to clear its expected sources.
    path = tmp_path / "questions.json"
    _write(
        path,
        [
            {
                "id": "q1",
                "query": "Who won the 1850 derby?",
                "user_tier": TIER_EMPLOYEE,
                "expected_answerable": False,
                "expected_source_paths": ["employee/derby.md"],
            }
        ],
    )

    with pytest.raises(ValueError, match="expected_source_paths"):
        load_questions(path)
