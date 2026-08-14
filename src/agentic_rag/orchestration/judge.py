import re

from agentic_rag.generation.llm_client import generate

_FIRST_WORD_RE = re.compile(r"[A-Za-z]+")


def classify_verdict(response: str) -> bool:
    """Fail-closed classification from a CLEAN-vs-flagged judge's raw
    response - shared by every local judge in this codebase
    (`check_for_injection` in `injection_judge.py`, `check_output_security`
    in `output_security.py`, `check_for_foul_language` in
    `foul_language.py`), each of which asks a differently-worded question
    of the same judge model but expects the identical single-word answer
    shape (CLEAN vs. one flagged keyword) and needs the identical parsing
    safety.

    Only the FIRST word is inspected, not a substring search across the
    whole response - a whole-response search misfires two ways: "unclean"
    contains "clean" (fails OPEN on a word that isn't the verdict), and a
    verbose CLEAN verdict that happens to repeat back a query term like
    "injection" (e.g. discussing a player's medical injection - a genuine
    football topic) would fail CLOSED on a legitimate query for the wrong
    reason. Both were found and verified live during self-review. Anything
    other than an unambiguous, leading CLEAN is treated as flagged - the
    caller decides what "flagged" means (injection, unsafe output, foul
    language), this parser only judges the verdict word.
    """
    match = _FIRST_WORD_RE.search(response)
    if match is None:
        return True

    first_word = match.group(0).upper()
    return first_word != "CLEAN"


def run_judge(
    prompt: str, *, model: str, base_url: str, timeout: int, temperature: float
) -> tuple[bool, str]:
    """Run the shared generate-then-classify skeleton every local judge in
    this codebase repeats: call the judge model with an already-built
    `prompt` and classify its verdict via `classify_verdict()`.

    Callers build their own judge-specific prompt (their own delimited
    template) and interpret the returned bool according to their own
    flagged-state meaning (injection, unsafe output, foul language) - this
    helper only executes the call-and-classify step, returning both the
    verdict and the raw response so callers can build their own result
    dataclass carrying the evidence alongside the verdict.

    Raises GenerationError if the LLM call itself fails - an
    infrastructure problem, not a judgment call this function should paper
    over by guessing either way.
    """
    response = generate(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    )
    return classify_verdict(response), response
