import json

from agentic_rag.evaluation.judge import FaithfulnessCheckResult
from agentic_rag.evaluation.report import EvalQuestionResult, build_report, report_to_json_dict


def _result(
    *,
    question_id="q1",
    expected_answerable=True,
    retrieval_hit=True,
    answered=True,
    faithfulness=None,
    hallucinated=False,
    error=None,
    duration_seconds=1.0,
):
    return EvalQuestionResult(
        question_id=question_id,
        query="q",
        expected_answerable=expected_answerable,
        expected_source_paths=["employee/a.md"],
        answer_text="the answer",
        cited_paths=["employee/a.md"],
        retrieval_hit=retrieval_hit,
        answered=answered,
        faithfulness=faithfulness,
        hallucinated=hallucinated,
        error=error,
        duration_seconds=duration_seconds,
    )


def test_build_report_computes_retrieval_precision_over_answerable_questions_only():
    results = [
        _result(question_id="q1", expected_answerable=True, retrieval_hit=True),
        _result(question_id="q2", expected_answerable=True, retrieval_hit=False),
        _result(question_id="q3", expected_answerable=False, retrieval_hit=None),
    ]

    report = build_report(results)

    assert report.retrieval_precision == 0.5


def test_build_report_computes_faithfulness_rate_over_judged_questions_only():
    results = [
        _result(
            question_id="q1",
            answered=True,
            faithfulness=FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN"),
        ),
        _result(
            question_id="q2",
            answered=True,
            faithfulness=FaithfulnessCheckResult(
                is_faithful=False, raw_judge_response="UNSUPPORTED"
            ),
        ),
        _result(question_id="q3", expected_answerable=False, answered=False, faithfulness=None),
    ]

    report = build_report(results)

    assert report.faithfulness_rate == 0.5


def test_build_report_computes_hallucination_rate_over_all_scored_questions():
    results = [
        _result(question_id="q1", hallucinated=False),
        _result(question_id="q2", hallucinated=True),
        _result(question_id="q3", hallucinated=False),
        _result(question_id="q4", hallucinated=False),
    ]

    report = build_report(results)

    assert report.hallucination_rate == 0.25


def test_build_report_handles_no_answerable_questions():
    results = [_result(question_id="q1", expected_answerable=False, retrieval_hit=None)]

    report = build_report(results)

    assert report.retrieval_precision is None


def test_build_report_handles_no_judged_questions():
    results = [
        _result(question_id="q1", expected_answerable=False, answered=False, faithfulness=None)
    ]

    report = build_report(results)

    assert report.faithfulness_rate is None


def test_build_report_preserves_the_per_question_results():
    results = [_result(question_id="q1"), _result(question_id="q2")]

    report = build_report(results)

    assert [r.question_id for r in report.results] == ["q1", "q2"]


def test_build_report_excludes_an_errored_question_from_every_metric():
    # An errored question's placeholder fields (hallucinated=False,
    # retrieval_hit=None, etc.) are not real measurements - counting them
    # would silently pull hallucination_rate down as if the system had
    # been proven not to hallucinate on a question it never got to answer.
    results = [
        _result(question_id="q1", retrieval_hit=True, hallucinated=False),
        _result(
            question_id="q2",
            retrieval_hit=None,
            answered=False,
            hallucinated=False,
            error="GenerationError: Ollama timed out",
        ),
    ]

    report = build_report(results)

    assert report.retrieval_precision == 1.0
    assert report.hallucination_rate == 0.0
    assert report.errored_count == 1


def test_build_report_hallucination_rate_is_none_when_every_question_errored():
    results = [
        _result(question_id="q1", answered=False, hallucinated=False, error="boom"),
    ]

    report = build_report(results)

    assert report.hallucination_rate is None
    assert report.errored_count == 1


def test_build_report_errored_count_is_zero_when_nothing_errored():
    results = [_result(question_id="q1")]

    report = build_report(results)

    assert report.errored_count == 0


def test_build_report_computes_average_duration_over_every_question():
    results = [
        _result(question_id="q1", duration_seconds=2.0),
        _result(question_id="q2", duration_seconds=4.0),
    ]

    report = build_report(results)

    assert report.average_duration_seconds == 3.0


def test_build_report_average_duration_includes_errored_questions():
    # Unlike the correctness metrics, timing is a real measurement even
    # for a question that ultimately failed - a slow failure is still a
    # data point about system health, so it isn't excluded the way a
    # placeholder hallucinated=False is.
    results = [
        _result(question_id="q1", duration_seconds=2.0),
        _result(question_id="q2", duration_seconds=6.0, answered=False, error="boom"),
    ]

    report = build_report(results)

    assert report.average_duration_seconds == 4.0


def test_build_report_average_duration_is_none_for_no_questions():
    # Matches the other three metrics' None-not-0.0 convention: "no
    # questions ran" and "every question took 0 seconds" must stay
    # distinguishable in the report.
    report = build_report([])

    assert report.average_duration_seconds is None


def test_report_to_json_dict_is_json_serializable():
    results = [
        _result(
            question_id="q1",
            faithfulness=FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN"),
        )
    ]
    report = build_report(results)

    payload = report_to_json_dict(report)

    # Round-trips through the stdlib json encoder with no custom encoder -
    # the whole point of a plain-dict report shape is that any consumer
    # (a CI job, a human, another script) can read it with nothing more
    # than json.loads().
    serialized = json.dumps(payload)
    assert json.loads(serialized)["retrieval_precision"] == 1.0


def test_report_to_json_dict_includes_the_per_question_breakdown():
    results = [
        _result(
            question_id="q1",
            faithfulness=FaithfulnessCheckResult(is_faithful=True, raw_judge_response="CLEAN"),
        )
    ]
    report = build_report(results)

    payload = report_to_json_dict(report)

    assert payload["results"][0]["question_id"] == "q1"
    assert payload["results"][0]["faithfulness"] == {
        "is_faithful": True,
        "raw_judge_response": "CLEAN",
    }


def test_report_to_json_dict_represents_a_none_faithfulness_as_null():
    results = [
        _result(question_id="q1", expected_answerable=False, answered=False, faithfulness=None)
    ]
    report = build_report(results)

    payload = report_to_json_dict(report)

    assert payload["results"][0]["faithfulness"] is None


def test_report_to_json_dict_includes_errored_count():
    results = [_result(question_id="q1", answered=False, error="boom")]
    report = build_report(results)

    payload = report_to_json_dict(report)

    assert payload["errored_count"] == 1
    assert payload["results"][0]["error"] == "boom"


def test_report_to_json_dict_includes_duration():
    results = [_result(question_id="q1", duration_seconds=3.5)]
    report = build_report(results)

    payload = report_to_json_dict(report)

    assert payload["average_duration_seconds"] == 3.5
    assert payload["results"][0]["duration_seconds"] == 3.5
