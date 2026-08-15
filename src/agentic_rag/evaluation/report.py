from __future__ import annotations

import dataclasses
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

    `error` is `None` for a normally-scored question. Set to a message
    when the question couldn't be scored at all (the pipeline itself
    raised - a judge-model timeout/OOM, an Ollama connection failure) -
    every other field is a placeholder in that case (`""`/`[]`/`False`/
    `None`), not a real measurement, and `build_report()` excludes an
    errored question from every *correctness* metric's denominator rather
    than letting a placeholder `False` silently count as a confirmed "not
    hallucinated" or similar.

    `duration_seconds` covers the whole per-question round trip -
    retrieval, generation, and (when it runs) the faithfulness judge call
    - measured regardless of whether the question errored, since "how
    long before it failed" is still a meaningful number, unlike the
    correctness metrics an error genuinely has nothing to say about.
    `build_report()` deliberately includes an errored question's
    `duration_seconds` in `average_duration_seconds` - the one field this
    exclusion does *not* apply to.
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
    error: str | None
    duration_seconds: float


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics plus the full per-question breakdown they were
    computed from, so a report is self-auditing - a reader can always
    trace a summary number back to the individual results it came from."""

    results: list[EvalQuestionResult]
    retrieval_precision: float | None
    faithfulness_rate: float | None
    hallucination_rate: float | None
    errored_count: int
    average_duration_seconds: float | None


def build_report(results: list[EvalQuestionResult]) -> EvaluationReport:
    """Aggregate per-question `results` into an `EvaluationReport`.

    `retrieval_precision`/`faithfulness_rate` are averaged only over
    questions where that metric actually applies (`retrieval_hit`/
    `faithfulness` is not `None`) - `None` (not `0.0`) if no question in
    this run had that metric apply at all, since "0% precision" and "no
    answerable questions were run" are different facts a report reader
    must not confuse.

    A question with `error` set is excluded from every correctness
    metric's denominator, not just skipped for the metrics that don't
    apply to it - its placeholder `hallucinated=False` is not a real
    measurement, and counting it would silently pull `hallucination_rate`
    down as if the system had been proven not to hallucinate on a
    question it never actually got to answer. `errored_count` surfaces
    how many questions were excluded this way, so a report reader isn't
    misled by a metric computed over fewer questions than the full run
    without knowing it.

    `average_duration_seconds` is averaged over *every* question,
    including errored ones - unlike the correctness metrics, timing is a
    real measurement even for a question that ultimately failed (a slow
    failure is still a data point about system health), so there's no
    equivalent reason to exclude it. It's `None` (not `0.0`) when
    `results` is empty, for the same reason `retrieval_precision` and the
    others are `None` rather than `0.0` in that case - "no questions ran"
    and "every question took 0 seconds" must stay distinguishable.
    """
    scored = [r for r in results if r.error is None]
    errored_count = len(results) - len(scored)

    retrieval_hits = [r.retrieval_hit for r in scored if r.retrieval_hit is not None]
    faithfulness_verdicts = [
        r.faithfulness.is_faithful for r in scored if r.faithfulness is not None
    ]

    return EvaluationReport(
        results=results,
        retrieval_precision=(sum(retrieval_hits) / len(retrieval_hits))
        if retrieval_hits
        else None,
        faithfulness_rate=(sum(faithfulness_verdicts) / len(faithfulness_verdicts))
        if faithfulness_verdicts
        else None,
        hallucination_rate=(sum(r.hallucinated for r in scored) / len(scored))
        if scored
        else None,
        errored_count=errored_count,
        average_duration_seconds=(sum(r.duration_seconds for r in results) / len(results))
        if results
        else None,
    )


def report_to_json_dict(report: EvaluationReport) -> dict:
    """Convert `report` into a plain dict of JSON-native types (no
    dataclasses, no custom encoder needed) - so any consumer (a CI job, a
    person reading the file, another script) can read it with nothing
    more than `json.loads()`.

    `dataclasses.asdict()` recurses through `EvalQuestionResult` and its
    nested `FaithfulnessCheckResult` (including the `None` case)
    automatically - hand-listing every field name here a second time
    would just be a second place field renames have to be remembered.
    """
    return {
        "retrieval_precision": report.retrieval_precision,
        "faithfulness_rate": report.faithfulness_rate,
        "hallucination_rate": report.hallucination_rate,
        "errored_count": report.errored_count,
        "average_duration_seconds": report.average_duration_seconds,
        "results": [dataclasses.asdict(r) for r in report.results],
    }
