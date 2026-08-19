# `orchestration/rewrite.py`

**Purpose:** This file makes multi-turn conversation work with the retrieval system. When a user asks a follow-up question like "what about his assists?" after previously asking about a specific player, that follow-up is meaningless on its own — the word "his" only makes sense in light of the earlier conversation. Since the retrieval/search step only ever sees a single query string (it has no memory of the conversation), this file's job is to rewrite such a follow-up into a single, self-contained question — e.g. turning "what about his assists?" into "What are [Player Name]'s assists?" — before it gets used for searching. This is what the project's requirements refer to as multi-turn context support (labeled "FR2" in the code's docstring).

## Line-by-line walkthrough

### Lines 1-3 — Imports
```python
from dataclasses import dataclass

from agentic_rag.generation.llm_client import GenerationError, generate
```
- `from dataclasses import dataclass` — imports the decorator used to define `ConversationTurn` below as a simple, structured data holder.
- `from agentic_rag.generation.llm_client import GenerationError, generate` — imports the shared function that calls the LLM (`generate`) and the exception type (`GenerationError`) this file raises when the rewrite fails — the same shared pieces used by `answer.py` and `decompose.py`.

### Lines 5-12 — Rewrite prompt template
```python
_REWRITE_PROMPT_TEMPLATE = """Given the conversation history below, rewrite the user's latest question into a single, self-contained question that can be understood without the history. Resolve any pronouns or references to earlier turns. Do not answer the question - only rewrite it. Reply with ONLY the rewritten question, nothing else.

Conversation history:
{history}

Latest question: {query}

Rewritten question:"""
```
- A template string with two placeholders, `{history}` and `{query}`. It instructs the model to rewrite (not answer) the latest question so it stands alone without needing the prior conversation, explicitly telling it to resolve pronouns ("his", "it", "them") and other references back to earlier turns, and to reply with nothing but the rewritten question itself.

### Lines 15-18 — `ConversationTurn` dataclass
```python
@dataclass(frozen=True)
class ConversationTurn:
    user_query: str
    assistant_answer: str
```
- `@dataclass(frozen=True)` — defines an immutable data class representing one exchange in the conversation.
- `user_query: str` — what the user asked in that turn.
- `assistant_answer: str` — what the assistant answered in that turn.
- Together, a list of these objects (`history`) represents the full back-and-forth of the conversation so far, which this file uses to give the model the context it needs to resolve references in the newest question.

### Lines 21-26 — `_format_history`
```python
def _format_history(history: list[ConversationTurn]) -> str:
    lines = []
    for turn in history:
        lines.append(f"User: {turn.user_query}")
        lines.append(f"Assistant: {turn.assistant_answer}")
    return "\n".join(lines)
```
- `def _format_history(...)` — a private helper that converts the list of `ConversationTurn` objects into a single readable text block for embedding in the prompt.
- `lines = []` — starts an empty list to collect formatted lines.
- The `for` loop appends two lines per turn: one prefixed `User:` with that turn's question, and one prefixed `Assistant:` with that turn's answer — reconstructing the conversation as a simple transcript.
- `"\n".join(lines)` — joins all the lines together with newlines into one multi-line string, which becomes the `{history}` portion of the prompt.

### Lines 29-79 — `rewrite_query`
```python
def rewrite_query(
    history: list[ConversationTurn],
    query: str,
    *,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
) -> str:
```
- The main function of this file. It takes the conversation `history` and the user's latest `query`, plus keyword-only parameters (`model`, `base_url`, `timeout`, `temperature`) controlling the LLM call, and returns a string: the rewritten, self-contained question.
- The docstring (lines 38-65, summarized) explains the reasoning behind several choices: rewriting exists so that retrieval — which has no awareness of conversation history — can still find the right information even for a follow-up question that only makes sense in context; when `history` is empty (this is the first turn of the conversation), the function returns the query unchanged with **no** LLM call at all, because a first turn is inherently already self-contained, and calling the model anyway would just add latency for no benefit; the function raises `GenerationError` if the LLM call fails or if it returns an empty/whitespace-only "rewrite," because silently falling back to either the raw, un-rewritten query or an unusable empty string would hide a real failure from the caller; and `temperature` is required with no default (matching the same "no defaults on config-mirroring parameters" convention used elsewhere in the codebase) because this call, unlike `decompose_query()` in `decompose.py` (which deliberately wants some variability across retries), is called exactly once per turn with no retry logic — so there is nothing to gain from letting its output vary, only correctness to lose, since the same conversation should always resolve a pronoun like "it" to the same self-contained question. The docstring also notes this was found to be a real, previously-unpinned bug discovered by self-review after similar temperature bugs were found and fixed in `generate_answer()` and `decompose_query()`.

```python
    if not history:
        return query
```
- If `history` is empty (an empty list is falsy in Python), the function immediately returns `query` unchanged, skipping the LLM call entirely — implementing the "first turn is already self-contained" behavior described in the docstring.

```python
    prompt = _REWRITE_PROMPT_TEMPLATE.format(
        history=_format_history(history), query=query
    )
    rewritten = generate(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    ).strip()
```
- Builds the actual prompt by filling in the template with the formatted conversation history (from `_format_history`) and the raw latest query.
- Calls the shared `generate()` function to send the prompt to the LLM, and immediately calls `.strip()` on the result to remove any leading/trailing whitespace the model's response might include.

```python
    if not rewritten:
        raise GenerationError("the LLM returned an empty rewritten query")

    return rewritten
```
- If, after stripping whitespace, the rewritten text is empty (the model returned nothing usable), the function raises `GenerationError` with an explanatory message rather than silently falling back to something potentially wrong or unusable.
- Otherwise, it returns the successfully rewritten, self-contained question to the caller.
