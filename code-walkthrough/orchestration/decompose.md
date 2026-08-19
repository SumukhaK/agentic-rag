# `orchestration/decompose.py`

**Purpose:** This file handles breaking a complex user question down into a list of smaller, simpler sub-questions that can each be answered independently by retrieval. For example, a question comparing two players' stats might get split into "What are player A's stats?" and "What are player B's stats?" This matters because the retrieval system works best when searching for one focused idea at a time; a single, compound question can confuse the search and cause the system to miss relevant information. The file asks the language model to do this splitting, then cleans up its output so that downstream code always receives a plain list of question strings, regardless of whether the model added bullets, numbers, or other formatting it was told not to use.

## Line-by-line walkthrough

### Lines 1-3 — Imports
```python
import re

from agentic_rag.generation.llm_client import GenerationError, generate
```
- `import re` — brings in the regular-expression module, used below to detect and strip list-formatting markers (like `1.` or `-`) that the model might add even though it's told not to.
- `from agentic_rag.generation.llm_client import GenerationError, generate` — imports the shared LLM-calling function `generate()` (same one used by `answer.py`), and `GenerationError`, an exception type this file raises when something goes wrong with generating sub-questions.

### Lines 5-9 — Decomposition prompt template
```python
_DECOMPOSE_PROMPT_TEMPLATE = """Break the following question into a list of simple, independently-answerable sub-questions. If the question is already simple and doesn't need breaking down, output it unchanged as the only line. Reply with ONLY the sub-questions, one per line, with no numbering, bullets, or extra commentary.

Question: {query}

Sub-questions:"""
```
- A template string with one placeholder, `{query}`. It instructs the model to produce one sub-question per line, with no numbering or bullets, and explicitly tells it that if the question is already simple, it should just return that question unchanged as a single line rather than forcing an artificial split.

### Line 11 — List-marker regex
```python
_LIST_MARKER_RE = re.compile(r"^(?:\d+[.)](?=\s|$)|[-*•])\s*")
```
- Compiles a regex used to strip a leading list marker from the start of a line, in case the model ignores the "no numbering, bullets" instruction. It matches either:
  - `\d+[.)](?=\s|$)` — one or more digits followed by a `.` or `)`, but only when that's immediately followed by whitespace or the end of the line (a lookahead, which doesn't consume characters itself). This is what recognizes things like `1.` or `2)` as enumeration markers.
  - or `[-*•]` — a single dash, asterisk, or bullet character, which are common bullet-list symbols.
- `\s*` at the end also consumes any following whitespace so the cleaned text doesn't start with a stray space.

### Lines 14-19 — `_clean_line`
```python
def _clean_line(line: str) -> str:
    # The digit-marker branch requires whitespace or end-of-line right
    # after the "N." / "N)", so it only matches an actual enumeration
    # prefix - not the start of a decimal number like "1.85 xG", which is
    # realistic sub-question content in this domain.
    return _LIST_MARKER_RE.sub("", line.strip())
```
- `def _clean_line(line: str) -> str:` — a private helper that takes one line of the model's raw response and strips any leading list-marker formatting from it.
- `line.strip()` — first removes leading/trailing whitespace from the line.
- `_LIST_MARKER_RE.sub("", ...)` — then removes a matching list-marker prefix, if present, replacing it with nothing.
- The comment above explains a subtle but important design detail: the regex's lookahead (`(?=\s|$)`) is what stops it from wrongly treating a decimal number as a list marker. In football statistics, a sub-question might legitimately start with something like "1.85 xG" (expected goals). Without the lookahead requiring whitespace or end-of-line right after the digit-dot, the regex could mistakenly strip the "1." off "1.85 xG" and corrupt real content. The lookahead ensures only an actual "N." or "N)" used as a list marker (followed by a space or nothing else) gets stripped.

### Lines 22-71 — `decompose_query`
```python
def decompose_query(
    query: str, *, model: str, base_url: str, timeout: int, temperature: float
) -> list[str]:
```
- The main function of this file. It takes the user's `query` string, plus keyword-only connection/behavior parameters (`model`, `base_url`, `timeout`, `temperature` — the `*` forces these to be passed by name), and returns a list of sub-question strings.
- The docstring (lines 25-56, summarized) explains several things: the result is always the LLM's response split line-by-line with list markers cleaned up; there's deliberately no separate "is this question complex?" pre-check step, because judging that would require the same kind of language understanding the LLM call already performs, so it would be redundant; the function raises `GenerationError` if the LLM call itself fails, or if after cleanup there are zero usable sub-questions left (e.g. the model returned only blank lines) — because silently returning an empty list would make any code that plans off this result behave as if there were nothing left to retrieve, with no clear error explaining why; and it explains why `temperature` has no default and must be passed explicitly. Notably, unlike `answer.py`'s `generate_answer()` and `rewrite.py`'s `rewrite_query()` (which always want `temperature=0.0` for consistency), this function's ideal temperature is *not* always zero: it's designed to be called with a low temperature on a first attempt (for best-effort determinism — live testing found that identical calls at the model's default, non-zero temperature could produce different sub-question splits, sometimes wrongly causing a simple question to be judged as needing more evidence than it does) but with a *higher* temperature on retries, because retrying with the same deterministic decomposition would just reproduce the same failed result — only exploring a genuinely different phrasing gives a retry a chance to succeed.

```python
    prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(query=query)
    response = generate(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    )
```
- Fills in the prompt template with the user's `query`.
- Calls the shared `generate()` function to send the prompt to the LLM and get back its raw text response.

```python
    sub_questions = [
        cleaned
        for line in response.splitlines()
        if (cleaned := _clean_line(line))
    ]
```
- `response.splitlines()` — breaks the model's response into individual lines.
- This is a list comprehension using the walrus operator (`:=`), which assigns the result of `_clean_line(line)` to `cleaned` *and* evaluates it in the same expression. For each line, `cleaned` is computed once, and the `if cleaned` condition filters out any line that becomes empty after cleaning (e.g. a blank line, or a line that was only a stray bullet character) — so `sub_questions` ends up as a list of only the non-empty, marker-stripped lines.

```python
    if not sub_questions:
        raise GenerationError("the LLM returned no usable sub-questions")

    return sub_questions
```
- If, after all cleanup, the list of sub-questions is empty, the function raises `GenerationError` with a descriptive message rather than silently returning an empty list — per the docstring's reasoning, letting this fail loudly instead of pretending there's simply nothing to retrieve.
- Otherwise, it returns the final list of cleaned sub-question strings to the caller.
