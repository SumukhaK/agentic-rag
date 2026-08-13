import re

from agentic_rag.generation.llm_client import GenerationError, generate

_DECOMPOSE_PROMPT_TEMPLATE = """Break the following question into a list of simple, independently-answerable sub-questions. If the question is already simple and doesn't need breaking down, output it unchanged as the only line. Reply with ONLY the sub-questions, one per line, with no numbering, bullets, or extra commentary.

Question: {query}

Sub-questions:"""

_LIST_MARKER_RE = re.compile(r"^(?:\d+[.)]|[-*•])\s*")


def _clean_line(line: str) -> str:
    return _LIST_MARKER_RE.sub("", line.strip()).strip()


def decompose_query(query: str, *, model: str, base_url: str, timeout: int) -> list[str]:
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
    """
    prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(query=query)
    response = generate(prompt, model=model, base_url=base_url, timeout=timeout)

    sub_questions = [
        cleaned
        for line in response.splitlines()
        if (cleaned := _clean_line(line))
    ]

    if not sub_questions:
        raise GenerationError("the LLM returned no usable sub-questions")

    return sub_questions
