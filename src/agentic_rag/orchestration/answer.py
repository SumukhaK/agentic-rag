from agentic_rag.generation.llm_client import generate
from agentic_rag.orchestration.planning import CANNOT_ANSWER_MESSAGE, PlanningResult
from agentic_rag.retrieval.search import SearchCandidate

_PROMPT_TEMPLATE = """Answer the question using ONLY the numbered sources below. Every factual claim must cite its source number in brackets, e.g. [1]. Do not use any knowledge beyond what is given in the sources. If the sources do not contain enough information to answer, respond with exactly this sentence and nothing else: "{fallback_message}"

Sources:
{sources}

Question: {query}

Answer:"""


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
    return "\n\n".join(
        f"[{index}] (source: {candidate.relative_path}, access: {candidate.access_tier})\n{candidate.text}"
        for index, candidate in enumerate(candidates, start=1)
    )


def generate_answer(
    planning_result: PlanningResult,
    *,
    query: str,
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Produce the final grounded answer for `query` from `planning_result`.

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
    """
    if not planning_result.sufficient:
        return planning_result.message

    candidates = _deduplicated_candidates(planning_result)
    prompt = _PROMPT_TEMPLATE.format(
        fallback_message=CANNOT_ANSWER_MESSAGE,
        sources=_format_sources(candidates),
        query=query,
    )
    return generate(prompt, model=model, base_url=base_url, timeout=timeout)
