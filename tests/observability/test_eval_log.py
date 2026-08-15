import io
import json
import logging
import os

from agentic_rag.observability.eval_log import (
    LOGGER_NAME,
    configure_eval_logging,
    log_evaluation_run,
)


def test_log_evaluation_run_emits_exactly_one_info_record(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_evaluation_run(
            retrieval_precision=1.0,
            faithfulness_rate=0.75,
            hallucination_rate=0.167,
            errored_count=0,
            average_duration_seconds=76.7,
            report_path="eval/results/eval-20260815T135022.json",
            run_duration_seconds=460.2,
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO


def test_log_evaluation_run_message_is_valid_json_with_expected_fields(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_evaluation_run(
            retrieval_precision=1.0,
            faithfulness_rate=0.75,
            hallucination_rate=0.167,
            errored_count=0,
            average_duration_seconds=76.7,
            report_path="eval/results/eval-20260815T135022.json",
            run_duration_seconds=460.2,
        )

    payload = json.loads(caplog.records[0].getMessage())

    assert payload["event"] == "evaluation_run"
    assert payload["retrieval_precision"] == 1.0
    assert payload["faithfulness_rate"] == 0.75
    assert payload["hallucination_rate"] == 0.167
    assert payload["errored_count"] == 0
    assert payload["report_id"] == "eval-20260815T135022"
    assert payload["average_duration_seconds"] == 76.7
    assert payload["report_path"] == "eval/results/eval-20260815T135022.json"
    assert payload["run_duration_seconds"] == 460.2
    assert "timestamp" in payload


def test_log_evaluation_run_allows_none_metrics_when_nothing_was_scored():
    # retrieval_precision/faithfulness_rate/hallucination_rate/
    # average_duration_seconds are all None-capable per report.py's own
    # build_report() (distinguishing "no data" from "0") - the log must
    # preserve that distinction, not coerce None into 0.0.
    stream = io.StringIO()
    configure_eval_logging(stream=stream)

    log_evaluation_run(
        retrieval_precision=None,
        faithfulness_rate=None,
        hallucination_rate=None,
        errored_count=6,
        average_duration_seconds=None,
        report_path="eval/results/eval-x.json",
        run_duration_seconds=12.0,
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["retrieval_precision"] is None
    assert payload["faithfulness_rate"] is None
    assert payload["hallucination_rate"] is None
    assert payload["average_duration_seconds"] is None


def test_log_evaluation_run_derives_report_id_from_a_nested_path():
    # report_id is a portable correlation key derived from Path(...).stem
    # - os.sep, not a hardcoded separator, matches whatever this test's
    # own platform actually uses (Path parses backslash as a separator
    # only on Windows), avoiding the exact "hardcoded separator vs. the
    # platform's real one" bug class this codebase has hit before
    # (evaluation/runner.py's own _normalize_path()).
    stream = io.StringIO()
    configure_eval_logging(stream=stream)

    report_path = os.sep.join(["eval", "results", "eval-20260815T153329.json"])
    log_evaluation_run(
        retrieval_precision=1.0,
        faithfulness_rate=1.0,
        hallucination_rate=0.0,
        errored_count=0,
        average_duration_seconds=10.0,
        report_path=report_path,
        run_duration_seconds=60.0,
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["report_id"] == "eval-20260815T153329"


def test_configure_eval_logging_writes_json_lines_to_the_given_stream():
    stream = io.StringIO()

    configure_eval_logging(stream=stream)
    log_evaluation_run(
        retrieval_precision=1.0,
        faithfulness_rate=1.0,
        hallucination_rate=0.0,
        errored_count=0,
        average_duration_seconds=10.0,
        report_path="eval/results/eval-y.json",
        run_duration_seconds=60.0,
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "evaluation_run"
