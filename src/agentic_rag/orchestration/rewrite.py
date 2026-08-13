from dataclasses import dataclass

from agentic_rag.generation.llm_client import GenerationError, generate

_REWRITE_PROMPT_TEMPLATE = """Given the conversation history below, rewrite the user's latest question into a single, self-contained question that can be understood without the history. Resolve any pronouns or references to earlier turns. Do not answer the question - only rewrite it. Reply with ONLY the rewritten question, nothing else.

Conversation history:
{history}

Latest question: {query}

Rewritten question:"""


@dataclass(frozen=True)
class ConversationTurn:
    user_query: str
    assistant_answer: str


def _format_history(history: list[ConversationTurn]) -> str:
    lines = []
    for turn in history:
        lines.append(f"User: {turn.user_query}")
        lines.append(f"Assistant: {turn.assistant_answer}")
    return "\n".join(lines)


def rewrite_query(
    history: list[ConversationTurn],
    query: str,
    *,
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Rewrite `query` into a single, self-contained question given prior
    conversation turns, so retrieval doesn't need conversational context to
    find the right chunks (FR2 - multi-turn context).

    Returns `query` unchanged, with no LLM call, when there's no history to
    incorporate - the first turn of a conversation is already
    self-contained by definition, and calling the LLM would be pure
    wasted latency.

    Raises GenerationError if the LLM call fails, or if it returns an
    empty/whitespace-only "rewrite" - either way, a failed rewrite must
    not silently fall back to the raw query (or to an unusable empty one)
    without the caller knowing.
    """
    if not history:
        return query

    prompt = _REWRITE_PROMPT_TEMPLATE.format(
        history=_format_history(history), query=query
    )
    rewritten = generate(prompt, model=model, base_url=base_url, timeout=timeout).strip()

    if not rewritten:
        raise GenerationError("the LLM returned an empty rewritten query")

    return rewritten
