import re
from dataclasses import dataclass

from agentic_rag.generation.llm_client import generate
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, PlanningResult
from agentic_rag.retrieval.search import SearchCandidate

_PROMPT_TEMPLATE = """Answer the question using ONLY the numbered sources below. Every factual claim must cite its source number in brackets, e.g. [1]. Do not use any knowledge beyond what is given in the sources. If the sources do not contain enough information to answer, respond with exactly this sentence and nothing else: "{fallback_message}"

Sources:
{sources}

Question: {query}

Answer:"""

_CITATION_RE = re.compile(r"\[(\d+)\]")


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


def _format_sources(candidates: list[SearchCandidate]) -> str:
    return "\n\n---\n\n".join(
        f"[{index}] (source: {candidate.relative_path}, chunk: {candidate.chunk_index}, access: {candidate.access_tier})\n{candidate.text}"
        for index, candidate in enumerate(candidates, start=1)
    )


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


def generate_answer(
    planning_result: PlanningResult,
    *,
    query: str,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
) -> AnswerResult:
    """Produce the final grounded answer for `query` from `planning_result`,
    as an `AnswerResult(text, citations)` - not a bare `str`. A bare string
    was the original return type; self-review of the `POST /query` PR
    found it left FR1 unsatisfiable for any real API caller, since the
    `[N]` markers in `text` are meaningless without the source metadata
    that only existed inside the now-discarded `PlanningResult`.

    If retrieval was insufficient, returns `planning_result.message` (the
    canonical fallback from REQUIREMENTS.md §8 rule 2) directly, with no LLM
    call - there is nothing to ground an answer in, so generating would
    only risk the model reaching for outside knowledge.

    Otherwise assembles a citation-numbered prompt from every candidate
    chunk found across all sub-questions (deduplicated - the same chunk can
    answer more than one sub-question) and calls `generate()`. The prompt
    itself carries all three grounding rules from §8 (cite sources, say the
    canonical fallback verbatim if the sources turn out insufficient, never
    use outside knowledge) as a second line of defense: `sufficient=True`
    is only a coarse retrieval-only signal (see `planning.py`), not a
    guarantee the evidence actually answers the question, so the model
    needs its own instruction to fall back honestly if it doesn't.

    Prompt-following is a "please" the model can ignore, not a guarantee -
    so the generated text is validated before being trusted: it must be
    the canonical fallback verbatim, or cite at least one source number
    that's actually in range. A citation to a source that doesn't exist is
    worse than no citation (§8 rule 1 has no exceptions, and a fabricated
    citation carries false authority), so an answer that fails this check
    is replaced with the canonical fallback rather than returned as-is.

    `temperature` is required (e.g. `Settings.generation_temperature`,
    default `0.0`), matching this codebase's "no defaults on
    config-mirroring parameters" convention - a separate setting from the
    Phase 6 judges' `judge_temperature`, not a copy of it, since the
    tradeoff differs: a judge's single-word verdict has no reason to vary,
    but a natural-language answer's phrasing plausibly could. It defaults
    to the same `0.0` anyway because live testing found this isn't just a
    style/variety question here: calling `generate_answer()` 3x with the
    identical, already-`sufficient` `PlanningResult` produced the correct,
    cited answer twice and the canonical fallback once - Ollama's default
    non-zero sampling made a *correctness*-relevant call inconsistent, the
    same class of bug already fixed for the judges, just discovered later
    because this call sat unexercised by a real caller until Phase 7's
    `POST /query` existed to run it live.

    `citations` lists only the sources `text` actually cites, in ascending
    order - not every candidate offered to the prompt. A candidate the
    model had access to but chose not to reference wasn't evidence for
    anything in the final answer, so reporting it as a citation would
    misrepresent what the answer is actually grounded in. Empty whenever
    `text` is the canonical fallback (nothing was answered, so nothing was
    cited) or retrieval never ran at all.
    """
    if not planning_result.sufficient:
        return AnswerResult(text=planning_result.message, citations=[])

    candidates = _deduplicated_candidates(planning_result)
    if not candidates:
        return AnswerResult(text=CANNOT_ANSWER_MESSAGE, citations=[])

    prompt = _PROMPT_TEMPLATE.format(
        fallback_message=CANNOT_ANSWER_MESSAGE,
        sources=_format_sources(candidates),
        query=query,
    )
    answer = generate(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    )
    if not _is_grounded(answer, len(candidates)):
        return AnswerResult(text=CANNOT_ANSWER_MESSAGE, citations=[])

    return AnswerResult(text=answer, citations=_citations_used(answer, candidates))
