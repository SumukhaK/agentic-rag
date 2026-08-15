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
):
    return EvalQuestionResult(
        question_id=question_id,
        query="q",
        expected_answerable=expected_answerable,
        expected_source_paths=["tier-1/a.md"],
        answer_text="the answer",
        cited_paths=["tier-1/a.md"],
        retrieval_hit=retrieval_hit,
        answered=answered,
        faithfulness=faithfulness,
        hallucinated=hallucinated,
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


def test_build_report_computes_hallucination_rate_over_all_questions():
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
    results = [_result(question_id="q1", expected_answerable=False, answered=False, faithfulness=None)]
    report = build_report(results)

    payload = report_to_json_dict(report)

    assert payload["results"][0]["faithfulness"] is None
