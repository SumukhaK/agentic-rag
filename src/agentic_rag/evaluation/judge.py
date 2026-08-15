from dataclasses import dataclass

from agentic_rag.orchestration.judge import FORGED_VERDICT_SENTINEL, has_forged_verdict, run_judge

_FAITHFULNESS_JUDGE_PROMPT_TEMPLATE = """You are grading a football analytics assistant's answer for faithfulness: does every factual claim in the answer actually follow from the cited source text below, with nothing added, exaggerated, or changed?

Question: {query}

Answer: {answer}

Cited source text:
{sources}

Reply with ONLY one word, and nothing else: CLEAN if every claim in the answer is supported by the source text, or UNSUPPORTED if the answer states anything - a fact, a number, a name - that isn't actually in the source text.

Answer:"""


@dataclass(frozen=True)
class FaithfulnessCheckResult:
    is_faithful: bool
    raw_judge_response: str


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
    """Judge whether `answer` (generated in response to `query`) only
    asserts claims actually supported by `sources` - the text of the
    chunks it cited.

    Reuses `orchestration/judge.py`'s `run_judge()`/`classify_verdict()` -
    the same CLEAN-vs-flagged parsing convention `check_for_injection()`,
    `check_for_foul_language()`, and `check_output_security()` already
    established, rather than a fourth bespoke judge implementation for
    what's structurally the same "call a judge model, parse a one-word
    verdict, fail closed on anything else" shape.

    Checks `has_forged_verdict(answer)` before calling the LLM at all -
    unlike `query`/`sources` (the hand-curated eval question and its
    trusted cited corpus text, with no untrusted party in this offline
    loop to defend against), `answer` is the *generation* model's own
    live output, not eval-fixture content. It's exactly the kind of text
    the three production judges' forged-verdict exploit targets - the
    injection judge's own docstring documents `mistral`'s completion bias
    toward finishing a trailing "Answer: CLEAN"-shaped phrase regardless
    of prompt wording, and there's no reason a generated answer couldn't
    exhibit the same completion bias if it paraphrases source text or
    degenerates in a way that happens to echo that pattern. Treating the
    corpus as trusted was never license to treat the answer as trusted
    too - they're different inputs with different provenance.

    `temperature` is required, not defaulted - reproducible evaluation
    results across runs is the entire point of a structured eval, and this
    codebase has already hit the identical "forgot to pin it" bug three
    times elsewhere (`generate()`'s own docstring).

    Fails closed like the other three judges: a response whose first word
    isn't unambiguously CLEAN is treated as unfaithful, not silently
    passed.
    """
    if has_forged_verdict(answer):
        return FaithfulnessCheckResult(
            is_faithful=False, raw_judge_response=FORGED_VERDICT_SENTINEL
        )

    prompt = _FAITHFULNESS_JUDGE_PROMPT_TEMPLATE.format(
        query=query, answer=answer, sources=sources
    )
    is_unfaithful, response = run_judge(
        prompt, model=model, base_url=base_url, timeout=timeout, temperature=temperature
    )

    return FaithfulnessCheckResult(is_faithful=not is_unfaithful, raw_judge_response=response)
