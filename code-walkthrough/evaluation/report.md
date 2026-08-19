# `evaluation/report.py`

**Purpose:** This file defines the data structures that hold the outcome of an evaluation run - both the result for a single question and the aggregated summary across all questions - plus the logic that turns a list of individual question results into that summary (`build_report`), and a helper to convert the summary into a plain dictionary suitable for writing out as JSON (`report_to_json_dict`). The key theme throughout this file is being careful about *when a metric doesn't apply* versus *when it applies and the answer is "zero" or "false"* - for example, distinguishing "no answerable questions were run, so retrieval precision is undefined" from "questions were run and retrieval precision really was 0%". Getting this distinction wrong would make the evaluation numbers misleading.

## Line-by-line walkthrough

### Lines 1-6 — Imports
```python
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from agentic_rag.evaluation.judge import FaithfulnessCheckResult
```
- `from __future__ import annotations` — defers evaluation of type hints, allowing modern-style hints (like `float | None`) to be used smoothly.
- `import dataclasses` — imports the whole `dataclasses` module (not just the decorator) because this file also needs `dataclasses.asdict()` later, a function that recursively converts a dataclass instance (and any nested dataclasses inside it) into a plain dictionary.
- `from dataclasses import dataclass` — imports the `@dataclass` decorator directly, used to define the two record types below.
- `from agentic_rag.evaluation.judge import FaithfulnessCheckResult` — imports the result type produced by the faithfulness judge (defined in `judge.py`), because each per-question result needs to store that judge's verdict.

### Lines 9-40 — `EvalQuestionResult` dataclass and its docstring
```python
@dataclass(frozen=True)
class EvalQuestionResult:
    """The real pipeline's outcome for one `EvalQuestion`, with everything
    ...
    correctness metrics an error genuinely has nothing to say about.
    `build_report()` deliberately includes an errored question's
    `duration_seconds` in `average_duration_seconds` - the one field this
    exclusion does *not* apply to.
    """
```
- `@dataclass(frozen=True)` — again, an immutable record type: once a question's outcome has been computed, it shouldn't be changed.
- `class EvalQuestionResult:` — represents everything measured about how the real pipeline handled one evaluation question, both for computing the aggregate metrics and for a human to audit a single question's result by hand.
- The docstring explains three important design decisions: (1) `retrieval_hit` and `faithfulness` are `None` - not `False` or a forced verdict - when the metric genuinely doesn't apply to a question (e.g. `retrieval_hit` is meaningless for a question that isn't expected to be answerable, and `faithfulness` is only judged if the system actually produced an answer). Forcing a value would silently skew the aggregate metrics rather than correctly excluding that question. (2) `error` is `None` for a normally-scored question, but set to an error message if the pipeline itself failed to run for that question (e.g. a judge-model timeout, an Ollama connection failure) - in that case every other field is just a placeholder, not a real measurement, and `build_report()` (below) will exclude such questions from every *correctness* metric. (3) `duration_seconds` covers the whole per-question round trip and is measured regardless of whether the question errored - because "how long before it failed" is still a meaningful data point - and `build_report()` deliberately keeps errored questions in the timing average even though it excludes them from the correctness metrics.

### Lines 42-53 — `EvalQuestionResult` fields
```python
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
```
- `question_id: str` — echoes back the `id` of the `EvalQuestion` this result belongs to, for cross-referencing.
- `query: str` — the original question text, kept alongside the result for easy human review without needing to look it up elsewhere.
- `expected_answerable: bool` — copied from the original question, so a reader of the result alone knows what was expected without needing the original fixture.
- `expected_source_paths: list[str]` — likewise, the ground-truth expected citations, copied for the same self-contained-auditability reason.
- `answer_text: str` — the actual text the pipeline generated for this question.
- `cited_paths: list[str]` — the document paths the generated answer actually cited.
- `retrieval_hit: bool | None` — whether the actual citations overlapped with the expected ones; `None` when this doesn't apply (question not expected to be answerable).
- `answered: bool` — whether the system produced a real answer (as opposed to falling back to "I don't know").
- `faithfulness: FaithfulnessCheckResult | None` — the faithfulness judge's verdict (see `judge.py`), or `None` if faithfulness wasn't checked for this question.
- `hallucinated: bool` — whether this question counts as a hallucination (see the detailed logic in `runner.py`'s `_run_question`).
- `error: str | None` — `None` normally, or an error description if this question's evaluation itself failed to run.
- `duration_seconds: float` — how long this question took to process end-to-end.

### Lines 56-67 — `EvaluationReport` dataclass
```python
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
```
- `@dataclass(frozen=True)` — again an immutable record; the finished object shouldn't be mutated after being built.
- `class EvaluationReport:` — represents the full output of one evaluation run: both the aggregate numbers and the underlying data they were computed from, so any summary statistic can always be traced back to the specific questions that produced it (the docstring calls this "self-auditing").
- `results: list[EvalQuestionResult]` — the full list of per-question results, kept alongside the summary.
- `retrieval_precision: float | None` — the fraction of applicable questions where the system cited (at least one of) the expected source documents; `None` if no question in this run had this metric apply.
- `faithfulness_rate: float | None` — the fraction of applicable questions whose answer was judged faithful to its cited sources; `None` under the same "doesn't apply" condition.
- `hallucination_rate: float | None` — the fraction of scored questions that counted as a hallucination.
- `errored_count: int` — how many questions couldn't be scored at all due to an infrastructure failure.
- `average_duration_seconds: float | None` — the mean time taken per question, across every question including errored ones.

### Lines 70-98 — `build_report` function signature and docstring
```python
def build_report(results: list[EvalQuestionResult]) -> EvaluationReport:
    """Aggregate per-question `results` into an `EvaluationReport`.
    ...
    the others are `None` rather than `0.0` in that case - "no questions ran"
    and "every question took 0 seconds" must stay distinguishable.
    """
```
- `def build_report(results: list[EvalQuestionResult]) -> EvaluationReport:` — takes the raw list of per-question results and computes the aggregate `EvaluationReport`.
- The docstring restates the core design principle of this file: `retrieval_precision` and `faithfulness_rate` are averaged only over questions where that specific metric actually applies (not `None`), and are `None` themselves (not `0.0`) if the metric never applied to any question in the run - because "0% precision" and "no answerable questions were run" are meaningfully different facts that a reader must not confuse. It also explains that any question with `error` set is excluded from every correctness metric's denominator (its placeholder `hallucinated=False` isn't a real measurement, and counting it would falsely make `hallucination_rate` look better), with `errored_count` surfacing how many questions were excluded this way so nobody is misled about the true sample size. Finally, `average_duration_seconds` is the one metric averaged over *every* question including errored ones, since timing is a real, meaningful measurement even for a question that ultimately failed.

### Lines 99-100 — Splitting scored vs. errored results
```python
    scored = [r for r in results if r.error is None]
    errored_count = len(results) - len(scored)
```
- `scored = [r for r in results if r.error is None]` — builds a list of only the results that completed successfully (no infrastructure error), which is the subset used for all correctness metrics.
- `errored_count = len(results) - len(scored)` — the count of results that were excluded because they errored, computed by subtracting the successfully-scored count from the total.

### Lines 102-105 — Collecting the applicable retrieval and faithfulness verdicts
```python
    retrieval_hits = [r.retrieval_hit for r in scored if r.retrieval_hit is not None]
    faithfulness_verdicts = [
        r.faithfulness.is_faithful for r in scored if r.faithfulness is not None
    ]
```
- `retrieval_hits = [r.retrieval_hit for r in scored if r.retrieval_hit is not None]` — from the successfully-scored questions, collects just the `retrieval_hit` values for the questions where retrieval precision actually applies (i.e., excludes questions where it's `None`).
- `faithfulness_verdicts = [r.faithfulness.is_faithful for r in scored if r.faithfulness is not None]` — similarly collects the `is_faithful` boolean out of each question's `FaithfulnessCheckResult`, but only for questions where faithfulness was actually judged (not `None`).

### Lines 107-122 — Constructing and returning the `EvaluationReport`
```python
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
```
- `results=results` — the object keeps the complete, un-filtered original list of per-question results (including errored ones), for full auditability.
- `retrieval_precision=(sum(retrieval_hits) / len(retrieval_hits)) if retrieval_hits else None` — computes the fraction of applicable questions with a retrieval hit (since `True`/`False` sum as `1`/`0` in Python, `sum(retrieval_hits)` counts the hits); if the list is empty (no question had this metric apply), the result is `None` rather than dividing by zero or defaulting to `0.0`.
- `faithfulness_rate=(sum(faithfulness_verdicts) / len(faithfulness_verdicts)) if faithfulness_verdicts else None` — the same pattern for the fraction of judged questions that were faithful.
- `hallucination_rate=(sum(r.hallucinated for r in scored) / len(scored)) if scored else None` — the fraction of successfully-scored questions that counted as hallucinations; `None` if there were no scored questions at all (e.g. every question errored).
- `errored_count=errored_count` — passes through the count computed earlier.
- `average_duration_seconds=(sum(r.duration_seconds for r in results) / len(results)) if results else None` — averages `duration_seconds` over *every* result (using `results`, not `scored`), matching the docstring's point that timing data remains meaningful even for failed questions; `None` only if `results` itself is empty (no questions ran at all).

### Lines 125-135 — `report_to_json_dict` function signature and docstring
```python
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
```
- `def report_to_json_dict(report: EvaluationReport) -> dict:` — converts a full `EvaluationReport` object into a plain Python dictionary made only of JSON-friendly types (numbers, strings, lists, dicts, `None`, booleans), so it can be serialized with the standard `json` module without needing a custom encoder.
- The docstring explains why `dataclasses.asdict()` is used for the `results` list specifically: it automatically recurses through nested dataclasses (here, each `EvalQuestionResult`, which itself contains a nested `FaithfulnessCheckResult` or `None`), so the code doesn't need to hand-list every field name a second time - which would create a second place that needs updating whenever a field is renamed.

### Lines 136-143 — Building the dictionary
```python
    return {
        "retrieval_precision": report.retrieval_precision,
        "faithfulness_rate": report.faithfulness_rate,
        "hallucination_rate": report.hallucination_rate,
        "errored_count": report.errored_count,
        "average_duration_seconds": report.average_duration_seconds,
        "results": [dataclasses.asdict(r) for r in report.results],
    }
```
- `"retrieval_precision": report.retrieval_precision`, `"faithfulness_rate": report.faithfulness_rate`, `"hallucination_rate": report.hallucination_rate`, `"errored_count": report.errored_count`, `"average_duration_seconds": report.average_duration_seconds` — each top-level summary metric is copied over by name into the output dictionary, unchanged (these are already JSON-native types: floats, `None`, and an int).
- `"results": [dataclasses.asdict(r) for r in report.results]` — converts each individual `EvalQuestionResult` (including its nested `FaithfulnessCheckResult`) into a plain dictionary via `dataclasses.asdict()`, producing a list of fully JSON-serializable per-question records.
