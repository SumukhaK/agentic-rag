from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.evaluation.judge import FaithfulnessCheckResult


@dataclass(frozen=True)
class EvalQuestionResult:
    """The real pipeline's outcome for one `EvalQuestion`, with everything
    needed to both compute the aggregate metrics below and audit a single
    question's result by hand.

    `retrieval_hit`/`faithfulness` are `None`, not `False`/a forced
    verdict, when the metric genuinely doesn't apply to this question -
    `retrieval_hit` only makes sense for an `expected_answerable` question,
    and `faithfulness` is only judged when the system actually produced an
    answer to judge. Forcing a value in either case would silently pull
    the aggregate metrics toward whichever value was chosen, rather than
    correctly excluding the question from that metric's denominator.
    """

    question_id: str
    query: str
    expected_answerable: bool
    expected_source_paths: list[str]
    answer_text: str
    cited_paths: list[str]
    retrieval_hit: bool | None
    answered: bool
    faithfulness: FaithfulnessCheckResult | None
    hallucinated: bool


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics plus the full per-question breakdown they were
    computed from, so a report is self-auditing - a reader can always
    trace a summary number back to the individual results it came from."""

    results: list[EvalQuestionResult]
    retrieval_precision: float | None
    faithfulness_rate: float | None
    hallucination_rate: float


def build_report(results: list[EvalQuestionResult]) -> EvaluationReport:
    """Aggregate per-question `results` into an `EvaluationReport`.

    `retrieval_precision`/`faithfulness_rate` are averaged only over
    questions where that metric actually applies (`retrieval_hit`/
    `faithfulness` is not `None`) - `None` (not `0.0`) if no question in
    this run had that metric apply at all, since "0% precision" and "no
    answerable questions were run" are different facts a report reader
    must not confuse. `hallucination_rate` always applies across every
    question, answerable or not - it's the one metric this evaluation
    exists to bound regardless of whether the corpus had an answer.
    """
    retrieval_hits = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    faithfulness_verdicts = [
        r.faithfulness.is_faithful for r in results if r.faithfulness is not None
    ]

    return EvaluationReport(
        results=results,
        retrieval_precision=(sum(retrieval_hits) / len(retrieval_hits))
        if retrieval_hits
        else None,
        faithfulness_rate=(sum(faithfulness_verdicts) / len(faithfulness_verdicts))
        if faithfulness_verdicts
        else None,
        hallucination_rate=(sum(r.hallucinated for r in results) / len(results))
        if results
        else 0.0,
    )


def report_to_json_dict(report: EvaluationReport) -> dict:
    """Convert `report` into a plain dict of JSON-native types (no
    dataclasses, no custom encoder needed) - so any consumer (a CI job, a
    person reading the file, another script) can read it with nothing
    more than `json.loads()`.
    """
    return {
        "retrieval_precision": report.retrieval_precision,
        "faithfulness_rate": report.faithfulness_rate,
        "hallucination_rate": report.hallucination_rate,
        "results": [
            {
                "question_id": r.question_id,
                "query": r.query,
                "expected_answerable": r.expected_answerable,
                "expected_source_paths": r.expected_source_paths,
                "answer_text": r.answer_text,
                "cited_paths": r.cited_paths,
                "retrieval_hit": r.retrieval_hit,
                "answered": r.answered,
                "faithfulness": {
                    "is_faithful": r.faithfulness.is_faithful,
                    "raw_judge_response": r.faithfulness.raw_judge_response,
                }
                if r.faithfulness is not None
                else None,
                "hallucinated": r.hallucinated,
            }
            for r in report.results
        ],
    }
