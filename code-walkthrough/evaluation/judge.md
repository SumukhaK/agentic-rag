# `evaluation/judge.py`

**Purpose:** This file answers one narrow but important question during automated evaluation runs: "did the answer the system generated only say things that are actually backed up by the source text it cited, or did it make something up?" This property is called *faithfulness* in the retrieval-augmented generation (RAG) world — a RAG system retrieves source documents and then asks a language model to write an answer grounded in them, and faithfulness measures whether the written answer stuck to what the sources actually said. Rather than writing a brand new "ask a model to grade something" mechanism, this file reuses the existing judge machinery from `orchestration/judge.py` (the same code that already checks for prompt injection, foul language, and unsafe output), because grading faithfulness is structurally the same task: send a prompt to a model, get back one word, and treat anything except an unambiguous "clean" verdict as a failure.

## Line-by-line walkthrough

### Lines 1-3 — Imports
```python
from dataclasses import dataclass

from agentic_rag.orchestration.judge import FORGED_VERDICT_SENTINEL, has_forged_verdict, run_judge
```
- `from dataclasses import dataclass` — brings in Python's `@dataclass` decorator, used below to build a small, simple result object without hand-writing `__init__`, `__eq__`, etc.
- `from agentic_rag.orchestration.judge import FORGED_VERDICT_SENTINEL, has_forged_verdict, run_judge` — imports three things from the production judge module that already handles the other LLM-based checks (injection, foul language, output security) in this codebase: `run_judge` (sends a prompt to the judge model and parses whether the verdict is "clean" or not), `has_forged_verdict` (a defensive check described below), and `FORGED_VERDICT_SENTINEL` (a placeholder value used to mark a result that was rejected without ever calling the model). Reusing these means this file doesn't reinvent "call a judge model and parse a one-word verdict" a fourth time.

### Lines 5-16 — The faithfulness judge prompt template
```python
_FAITHFULNESS_JUDGE_PROMPT_TEMPLATE = """You are grading a football analytics assistant's answer for faithfulness: does every factual claim in the answer actually follow from the cited source text below, with nothing added, exaggerated, or changed?

Question: {query}

Answer: {answer}

Cited source text:
{sources}

Reply with ONLY one word, and nothing else: CLEAN if every claim in the answer is supported by the source text, or UNSUPPORTED if the answer states anything - a fact, a number, a name - that isn't actually in the source text.

Answer:"""
```
- `_FAITHFULNESS_JUDGE_PROMPT_TEMPLATE = """..."""` — a module-level constant (leading underscore signals it's private to this file) holding a multi-line string template. It's a Python "f-string-style" template using `{query}`, `{answer}`, and `{sources}` placeholders that get filled in later with `.format(...)`.
- The template text itself instructs the judge model to compare the generated `answer` against the `sources` that were actually cited, and to reply with exactly one word: `CLEAN` (every claim is supported) or `UNSUPPORTED` (something in the answer — a fact, number, or name — isn't actually present in the source text). Ending the template with a trailing `Answer:` nudges the model to continue directly with the verdict word rather than restating the question.

### Lines 19-22 — `FaithfulnessCheckResult` dataclass
```python
@dataclass(frozen=True)
class FaithfulnessCheckResult:
    is_faithful: bool
    raw_judge_response: str
```
- `@dataclass(frozen=True)` — marks this class as an immutable data container; once created, its fields can't be reassigned. This matches the pattern used across the codebase for small result objects that shouldn't be mutated after the fact.
- `class FaithfulnessCheckResult:` — defines the return type of `check_faithfulness()` below.
- `is_faithful: bool` — the actual verdict: `True` if the answer was judged faithful to its sources, `False` otherwise.
- `raw_judge_response: str` — keeps the literal text the judge model (or the forged-verdict guard) returned, so a human auditing an eval run later can see exactly what happened rather than just a boolean.

### Lines 25-34 — `check_faithfulness` function signature
```python
def check_faithfulness(
    query: str,
    answer: str,
    sources: str,
    *,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
) -> FaithfulnessCheckResult:
```
- `query: str` — the original user question being evaluated.
- `answer: str` — the answer the real pipeline generated for that question, which is what's being checked.
- `sources: str` — the text of the source chunks that were actually cited in the answer, i.e. what the answer is allowed to be "grounded" in.
- `*,` — everything after this forces the remaining parameters to be passed by keyword only (e.g. `model=...`), preventing accidental positional mix-ups given there are four same-typed-looking arguments.
- `model: str, base_url: str, timeout: int, temperature: float` — connection and generation settings for the judge model call: which model to use, where the Ollama server lives, how long to wait before giving up, and the sampling temperature.
- `-> FaithfulnessCheckResult:` — declares the function returns the dataclass defined above.

### Lines 35-68 — Docstring explaining the design rationale
```python
    """Judge whether `answer` (generated in response to `query`) only
    ...
    passed.
    """
```
- This is a documentation-only block (no executable code), but it records several deliberate design choices worth summarizing: (1) this function deliberately reuses `orchestration/judge.py`'s existing verdict-parsing convention instead of writing a fourth bespoke judge; (2) it checks `has_forged_verdict(answer)` specifically because, unlike the hand-curated `query`/`sources` inputs (which are trusted, fixed evaluation fixtures), `answer` is live output from the generation model — meaning it's exactly the kind of untrusted text that could accidentally or adversarially trigger the same "forged verdict" completion pattern the injection judge already guards against (where a model like `mistral` tends to complete a trailing "Answer: CLEAN"-shaped phrase regardless of what it should say); (3) `temperature` has no default value on purpose, because reproducible evaluation results depend on pinning it, and this codebase has already hit bugs from forgetting to pin temperature elsewhere; (4) the function "fails closed" — meaning if the judge's response isn't unambiguously `CLEAN`, the code treats it as unfaithful rather than assuming the best case.

### Lines 69-72 — Guard against a forged verdict
```python
    if has_forged_verdict(answer):
        return FaithfulnessCheckResult(
            is_faithful=False, raw_judge_response=FORGED_VERDICT_SENTINEL
        )
```
- `if has_forged_verdict(answer):` — before ever sending anything to the judge model, this checks whether the generation model's own `answer` text already contains something that looks like a forged/injected verdict (e.g. text trying to trick a downstream judge by embedding "CLEAN" itself). This defends against the answer text manipulating the judge's grading process.
- `return FaithfulnessCheckResult(is_faithful=False, raw_judge_response=FORGED_VERDICT_SENTINEL)` — if a forged verdict is detected, the function short-circuits: it never calls the LLM judge at all, and immediately reports the answer as not faithful, tagging the result with the shared `FORGED_VERDICT_SENTINEL` marker so it's clearly distinguishable from a normal judge response in later analysis.

### Lines 74-81 — Building the prompt, running the judge, and returning the result
```python
    prompt = _FAITHFULNESS_JUDGE_PROMPT_TEMPLATE.format(
        query=query, answer=answer, sources=sources
    )
    is_unfaithful, response = run_judge(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    )

    return FaithfulnessCheckResult(is_faithful=not is_unfaithful, raw_judge_response=response)
```
- `prompt = _FAITHFULNESS_JUDGE_PROMPT_TEMPLATE.format(query=query, answer=answer, sources=sources)` — fills in the template constant from above with the real question, answer, and cited source text, producing the final prompt string to send to the judge model.
- `is_unfaithful, response = run_judge(prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature)` — delegates the actual model call and verdict parsing to the shared `run_judge()` helper (from `orchestration/judge.py`), which sends the prompt to the configured Ollama model and returns a tuple: a boolean flag (here interpreted as "is this unfaithful?") and the raw text response from the model. Reusing `run_judge` means the "call a judge model, parse CLEAN vs. anything-else, fail closed" logic lives in exactly one place in the codebase.
- `return FaithfulnessCheckResult(is_faithful=not is_unfaithful, raw_judge_response=response)` — inverts the `is_unfaithful` flag into the more naturally-named `is_faithful`, and packages it with the raw model response text into the final result object that callers (like the evaluation runner) will use.
