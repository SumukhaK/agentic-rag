# `orchestration/answer.py`

**Purpose:** This file turns the results of retrieval (the chunks of football data the system found) into the final answer text the user actually sees. Its central job is to make sure the assistant never just makes things up: it builds a prompt that forces the language model to cite which numbered source backs up each claim, and then it double-checks the model's response before trusting it — throwing away any answer that cites a source that doesn't exist, or that doesn't cite anything at all. This is the last line of defense for the project's "the assistant never invents facts" rule, because a model can always ignore instructions in a prompt, so the code verifies the output rather than assuming the model followed orders.

## Line-by-line walkthrough

### Lines 1-6 — Imports
```python
import re
from dataclasses import dataclass

from agentic_rag.generation.llm_client import generate
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, PlanningResult
from agentic_rag.retrieval.search import SearchCandidate
```
- `import re` — brings in Python's regular-expression module, used later to find citation markers like `[1]` in the model's answer text.
- `from dataclasses import dataclass` — imports the decorator used to build simple, immutable data-holder classes (`Citation`, `AnswerResult`) without hand-writing constructors.
- `from agentic_rag.generation.llm_client import generate` — imports the shared function that actually sends a prompt to the language model (Ollama) and gets text back; this file doesn't talk to the model directly, it delegates through this shared client.
- `from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, PlanningResult` — `PlanningResult` is the object produced by an earlier stage (query planning/retrieval) that this file consumes as input; `CANNOT_ANSWER_MESSAGE` is the exact fallback sentence the system must say verbatim when it can't ground an answer, imported here so both the planning stage and this file always use the identical wording.
- `from agentic_rag.retrieval.search import SearchCandidate` — `SearchCandidate` is the type representing one retrieved chunk of text (with its file path, chunk index, and access tier); this file works with lists of these.

### Lines 8-15 — The answer-generation prompt template
```python
_PROMPT_TEMPLATE = """Answer the question using ONLY the numbered sources below. Every factual claim must cite its source number in brackets, e.g. [1]. Do not use any knowledge beyond what is given in the sources. If the sources do not contain enough information to answer, respond with exactly this sentence and nothing else: "{fallback_message}"

Sources:
{sources}

Question: {query}

Answer:"""
```
- This is a multi-line string template with three placeholders (`{fallback_message}`, `{sources}`, `{query}`) that get filled in later with `.format(...)`. It instructs the model to answer only from the given sources, to cite every claim with a bracketed source number, and to reply with the exact fallback sentence if the sources aren't enough — this is the "first line of defense" (the code-level checks later are the second line, since the model might not obey these instructions).

### Line 17 — Citation-number regex
```python
_CITATION_RE = re.compile(r"\[(\d+)\]")
```
- Compiles a regular expression that matches a bracketed number like `[1]` or `[23]` and captures the digits. Pre-compiling it once at module load time (rather than inside a function) is more efficient since it's reused by multiple functions below.

### Lines 20-31 — `Citation` dataclass
```python
@dataclass(frozen=True)
class Citation:
    """One source `generate_answer()`'s answer text actually cited -
    FR1's "document + exact chunk" (`docs/REQUIREMENTS.md` §8 rule 1),
    resolvable by a caller that only has the answer text and its `[N]`
    markers, not the `PlanningResult` that produced it."""

    number: int
    relative_path: str
    chunk_index: int
    access_tier: str
```
- `@dataclass(frozen=True)` — marks this as an immutable data class: once created, a `Citation` instance's fields can't be reassigned, which prevents accidental mutation after the answer has been finalized.
- `class Citation:` — represents one specific citation that appeared in the final answer text, resolved back to the actual document it points to.
- The docstring explains why this exists: a caller who only has the raw answer text (with `[1]`-style markers) has no way to know which document `[1]` refers to unless something records that mapping — this class is that record, satisfying a requirement (referred to as "FR1") that citations must be traceable to a specific document and chunk, not just a number.
- `number: int` — the citation number as it appeared in the text (e.g. `1`).
- `relative_path: str` — the file path of the source document, relative to some project root.
- `chunk_index: int` — which chunk of that document (documents are split into chunks for retrieval) this citation came from.
- `access_tier: str` — a permission/classification tag on the source (e.g. public vs restricted), carried through so a caller can know what tier of data was actually used.

### Lines 33-44 — `AnswerResult` dataclass
```python
@dataclass(frozen=True)
class AnswerResult:
    """`generate_answer()`'s return value: the answer text plus the
    citations it actually used, so a caller can resolve `[1]` to an actual
    document instead of just seeing an unexplained bracketed number - the
    gap self-review of the `POST /query` PR found (`PROJECT_TRACKER.md`'s
    Phase 7 log): `answer_with_cache()` used to return a bare `str`,
    discarding every candidate's `relative_path`/`chunk_index`/
    `access_tier` once the answer was produced."""

    text: str
    citations: list[Citation]
```
- Another frozen (immutable) dataclass, this is the overall return type of the main function in this file, `generate_answer()`.
- The docstring records a real design history: an earlier version of the code returned just a plain string (`str`), which meant all the citation metadata was thrown away right after the answer was generated. This class fixes that by bundling the answer text together with the list of citations it actually used.
- `text: str` — the final answer text shown to the user (either a real cited answer or the canonical "cannot answer" fallback).
- `citations: list[Citation]` — the list of `Citation` objects the answer actually referenced, in order.

### Lines 47-61 — `_deduplicated_candidates`
```python
def _deduplicated_candidates(planning_result: PlanningResult) -> list[SearchCandidate]:
    """Flatten candidates across every sub-question's outcome, keeping the
    first occurrence of each (relative_path, chunk_index) - the same chunk
    can be relevant evidence for more than one sub-question, and sending it
    to the prompt twice would waste context and let it get cited under two
    different numbers."""
    seen: set[tuple[str, int]] = set()
    deduplicated: list[SearchCandidate] = []
    for outcome in planning_result.outcomes:
        for candidate in outcome.candidates:
            key = (candidate.relative_path, candidate.chunk_index)
            if key not in seen:
                seen.add(key)
                deduplicated.append(candidate)
    return deduplicated
```
- `def _deduplicated_candidates(...)` — a private helper (leading underscore means it's only meant to be used inside this file) that takes the full `PlanningResult` (which holds one "outcome" per sub-question the original query was broken into) and returns a single flat list of unique source chunks.
- `seen: set[tuple[str, int]] = set()` — creates an empty set to track which `(relative_path, chunk_index)` pairs have already been added, so duplicates can be detected in constant time.
- `deduplicated: list[SearchCandidate] = []` — the output list being built up, preserving first-seen order.
- `for outcome in planning_result.outcomes:` — iterates over each sub-question's retrieval outcome.
- `for candidate in outcome.candidates:` — iterates over each retrieved chunk within that outcome.
- `key = (candidate.relative_path, candidate.chunk_index)` — builds the unique identity for a chunk: which file it came from and which chunk number within that file — the same chunk could otherwise be retrieved separately for two different sub-questions.
- `if key not in seen:` / `seen.add(key)` / `deduplicated.append(candidate)` — only keeps the first time a given chunk is encountered; skips later duplicates. The comment explains why this matters: sending the same chunk twice would waste prompt space and, worse, let the same evidence get two different citation numbers, which is confusing and wasteful.
- `return deduplicated` — hands back the flat, de-duplicated list in the order chunks were first seen.

### Lines 64-68 — `_format_sources`
```python
def _format_sources(candidates: list[SearchCandidate]) -> str:
    return "\n\n---\n\n".join(
        f"[{index}] (source: {candidate.relative_path}, chunk: {candidate.chunk_index}, access: {candidate.access_tier})\n{candidate.text}"
        for index, candidate in enumerate(candidates, start=1)
    )
```
- `def _format_sources(...)` — turns the list of candidate chunks into the block of text that gets embedded into the prompt under "Sources:".
- `enumerate(candidates, start=1)` — numbers each candidate starting at 1 (not 0), because the citation format used everywhere in this file (`[1]`, `[2]`, ...) is 1-indexed to match natural human counting and the model's own citation convention.
- For each candidate, it builds a string showing the citation number, the file path, chunk index, and access tier as a header line, followed by the chunk's actual text (`candidate.text`) on the next line — this is what lets the model both cite the number and read the content.
- `"\n\n---\n\n".join(...)` — joins all these per-source blocks together, separated by a blank-line-dash-blank-line divider, so the model can visually tell where one source ends and the next begins.

### Lines 71-83 — `_is_grounded`
```python
def _is_grounded(answer: str, source_count: int) -> bool:
    """An answer counts as grounded only if it cites at least one source
    and every citation number it uses actually refers to a source that was
    given - a citation is worse than none at all if it's fabricated, since
    it carries false authority (§8 rule 1 has "no exceptions"). The
    canonical fallback message is exempt: it makes no factual claim, so it
    needs no citation."""
    if answer.strip() == CANNOT_ANSWER_MESSAGE:
        return True
    citation_numbers = [int(match) for match in _CITATION_RE.findall(answer)]
    if not citation_numbers:
        return False
    return all(1 <= number <= source_count for number in citation_numbers)
```
- This function is the core safety check: it decides whether a model-generated answer is trustworthy ("grounded" means every factual claim is backed by a real, given source) before the code returns it to the user.
- `if answer.strip() == CANNOT_ANSWER_MESSAGE: return True` — if the model correctly replied with the exact canonical fallback sentence (after trimming whitespace), that's automatically considered grounded, because saying "I can't answer" makes no factual claim and therefore needs no citation to back it up.
- `citation_numbers = [int(match) for match in _CITATION_RE.findall(answer)]` — uses the regex from line 17 to find every `[N]`-style marker in the answer text and converts each captured digit string to an integer.
- `if not citation_numbers: return False` — if the answer contains zero citations (and isn't the fallback message), it's rejected as ungrounded — an uncited claim can't be trusted to come from the sources.
- `return all(1 <= number <= source_count for number in citation_numbers)` — checks that every citation number found is within the valid range of actual sources provided (from 1 up to how many candidates were given). If the model cites, say, `[7]` but only 5 sources existed, that's a fabricated/hallucinated citation, and this returns `False`. The docstring stresses that a fake citation is worse than none, because it looks authoritative while being false.

### Lines 86-102 — `_citations_used`
```python
def _citations_used(answer: str, candidates: list[SearchCandidate]) -> list[Citation]:
    """Resolve every citation number the answer actually references (not
    every candidate offered - a candidate the model chose not to cite
    wasn't evidence for anything in the final text) into a `Citation`, in
    ascending order. Only called after `_is_grounded()` has already
    confirmed every number in `answer` is `1..len(candidates)`, so the
    `candidates[number - 1]` index is always in range here."""
    numbers = sorted({int(match) for match in _CITATION_RE.findall(answer)})
    return [
        Citation(
            number=number,
            relative_path=candidates[number - 1].relative_path,
            chunk_index=candidates[number - 1].chunk_index,
            access_tier=candidates[number - 1].access_tier,
        )
        for number in numbers
    ]
```
- This function converts the raw `[N]` markers found in an already-validated answer into a clean list of `Citation` objects with full source metadata.
- `numbers = sorted({int(match) for match in _CITATION_RE.findall(answer)})` — finds all citation numbers in the answer, puts them in a set to remove duplicates (if `[1]` appears twice, it should only produce one `Citation`), then sorts them in ascending order for a predictable, readable result.
- The list comprehension builds one `Citation` per unique number: `candidates[number - 1]` converts the 1-indexed citation number back to a 0-indexed list position to look up the matching `SearchCandidate`, then copies its `relative_path`, `chunk_index`, and `access_tier` into the new `Citation`.
- The docstring notes an important safety detail: this function trusts that every number is in-range only because it's *always* called after `_is_grounded()` has already verified that — so the `candidates[number - 1]` lookup can't go out of bounds here.

### Lines 105-186 — `generate_answer` (the main entry point)
```python
def generate_answer(
    planning_result: PlanningResult,
    *,
    query: str,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
) -> AnswerResult:
```
- This is the function other modules call to produce the final answer. Its signature takes the `PlanningResult` (containing retrieved evidence) as the first positional argument, then a set of keyword-only arguments (`*` forces everything after it to be passed by name, not position, to avoid mistakes like swapping `model` and `base_url`): `query` (the user's question), `model`/`base_url`/`timeout` (how to reach the LLM), and `temperature` (how random the model's output should be).
- The long docstring (lines 114-166, summarized rather than quoted here) explains several design decisions: it documents that this function used to return a bare string and that discarding citation metadata was a real bug found during self-review of a pull request; it explains that when retrieval didn't find enough evidence, no LLM call is made at all — there's nothing to ground an answer in, so calling the model would only risk it inventing an answer from outside knowledge; it explains that the prompt's own instructions (cite sources, use the fallback sentence verbatim, don't use outside knowledge) are a "second line of defense," not a guarantee, because the model can ignore instructions — which is exactly why `_is_grounded()` exists to check the actual output afterward; and it explains why `temperature` has no default value and is required to be passed in explicitly (usually `0.0`) — because live testing showed that calling this function multiple times with an identical, already-sufficient input could produce different answers (sometimes cited correctly, sometimes falling back unnecessarily) when the model's sampling temperature wasn't pinned to zero, which is a real correctness bug, not just stylistic variation.

```python
    if not planning_result.sufficient:
        return AnswerResult(text=planning_result.message, citations=[])
```
- If the planning/retrieval stage already determined the evidence gathered was **not** sufficient to answer the question (`planning_result.sufficient` is `False`), this function immediately returns that stage's own fallback message with no citations and, crucially, without calling the LLM at all — since there's nothing to ground an answer in.

```python
    candidates = _deduplicated_candidates(planning_result)
    if not candidates:
        return AnswerResult(text=CANNOT_ANSWER_MESSAGE, citations=[])
```
- Otherwise, it flattens and de-duplicates all the retrieved chunks using the helper from lines 47-61.
- As a defensive check, if that somehow yields an empty list (e.g. `sufficient` was `True` but no actual candidates exist), it falls back to the canonical "cannot answer" message rather than trying to build a prompt with no sources.

```python
    prompt = _PROMPT_TEMPLATE.format(
        fallback_message=CANNOT_ANSWER_MESSAGE,
        sources=_format_sources(candidates),
        query=query,
    )
    answer = generate(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    )
```
- Builds the actual prompt text by filling in the template from lines 8-15 with the canonical fallback sentence, the formatted sources block (from `_format_sources`), and the user's question.
- Calls the shared `generate()` function (imported at the top) to send this prompt to the LLM and get back the raw answer text, passing through all the connection/behavior parameters (`model`, `base_url`, `timeout`, `temperature`).

```python
    if not _is_grounded(answer, len(candidates)):
        return AnswerResult(text=CANNOT_ANSWER_MESSAGE, citations=[])

    return AnswerResult(text=answer, citations=_citations_used(answer, candidates))
```
- Runs the grounding check from lines 71-83 on the model's raw answer, passing how many real sources were available. If the answer fails this check (no citations, or a citation pointing outside the valid range), the function discards the model's answer entirely and substitutes the canonical "cannot answer" message instead — treating an ungrounded answer as no better than no answer, per the project's "no exceptions" citation rule.
- If the answer passes the check, the function returns the real answer text along with the resolved list of `Citation` objects built by `_citations_used()`, giving the caller both the human-readable text and the structured evidence trail behind it.
