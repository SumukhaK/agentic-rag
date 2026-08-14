import re

from agentic_rag.generation.llm_client import GenerationError, generate

_DECOMPOSE_PROMPT_TEMPLATE = """Break the following question into a list of simple, independently-answerable sub-questions. If the question is already simple and doesn't need breaking down, output it unchanged as the only line. Reply with ONLY the sub-questions, one per line, with no numbering, bullets, or extra commentary.

Question: {query}

Sub-questions:"""

_LIST_MARKER_RE = re.compile(r"^(?:\d+[.)](?=\s|$)|[-*•])\s*")


def _clean_line(line: str) -> str:
    # The digit-marker branch requires whitespace or end-of-line right
    # after the "N." / "N)", so it only matches an actual enumeration
    # prefix - not the start of a decimal number like "1.85 xG", which is
    # realistic sub-question content in this domain.
    return _LIST_MARKER_RE.sub("", line.strip())


def decompose_query(
    query: str, *, model: str, base_url: str, timeout: int, temperature: float
) -> list[str]:
    """Break `query` into independently-answerable sub-questions, one per
    line of the LLM's response, with common list markers (numbering,
    bullets) stripped in case the model ignores the "no numbering"
    instruction.

    If `query` is already simple/atomic, the LLM is instructed to return
    it unchanged as the only line, so the result is a single-item list
    equal to the original question - there's no separate "is this
    complex" pre-check, since that judgment itself requires the same kind
    of understanding the LLM call already provides.

    Raises GenerationError if the LLM call fails, or if it returns no
    usable sub-questions after cleanup (e.g. an empty/whitespace-only
    response) - a decomposition that silently produces zero sub-questions
    would make anything that plans off the result act as if there were
    nothing left to retrieve, without any error to explain why.

    `temperature` is required (e.g. `Settings.decompose_temperature` /
    `Settings.decompose_retry_temperature` - see `plan_and_retrieve()`'s
    docstring, `planning.py`) with no default, matching this codebase's
    "no defaults on config-mirroring parameters" convention. Unlike
    `generate_answer()`/`rewrite_query()`, this call's ideal temperature
    is *not* always `0.0`: `plan_and_retrieve()` deliberately calls this
    function with a low temperature on the first attempt (best-effort
    determinism - live-tested to matter: 3 identical calls at Ollama's
    default temperature produced 3 different sub-question pairs, one of
    which caused a query a single decomposition answers correctly to be
    misjudged `sufficient=False`) and a higher temperature on retries
    (retrying with the *same* deterministic decomposition would just
    reproduce the same "insufficient" result - a retry only has a chance
    of succeeding if it explores a genuinely different phrasing).
    """
    prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(query=query)
    response = generate(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    )

    sub_questions = [
        cleaned
        for line in response.splitlines()
        if (cleaned := _clean_line(line))
    ]

    if not sub_questions:
        raise GenerationError("the LLM returned no usable sub-questions")

    return sub_questions
