# Evaluation

This directory holds the structured evaluation for the football intelligence
assistant's retrieval-augmented generation (RAG) pipeline: a small, hand-curated
question set answered through the real `POST /query` code path, scored against
ground truth. It exists to answer one question with real numbers instead of a
gut feeling: **is the assistant grounded, or is it making things up?**

Full design rationale and the self-review history live in
[`PROJECT_TRACKER.md`](../PROJECT_TRACKER.md), Phase 8. This file is the
plain-language reference: what's being tested, what each metric means, what
the latest run actually produced, and what that says about the system.

## How to run it

```bash
python -m agentic_rag.evaluation.runner
```

This indexes `eval/corpus/` into a dedicated Qdrant collection (kept
completely separate from any production data), answers every question in
`eval/questions.json` through the real pipeline, and writes a timestamped
JSON report to `eval/results/` (gitignored — it's run output, not a fixture).
It also prints the summary metrics to stdout.

## The corpus

`eval/corpus/` holds 4 small, real-football documents, split across the two
access tiers the pipeline actually enforces:

| File | Tier | Content |
|---|---|---|
| `tier-1/derby.md` | tier-1 | Arsenal beat Tottenham 3-1 in the North London Derby (15 Mar 2025); Saka scored twice, Ødegaard once, Son for Spurs |
| `tier-1/transfer.md` | tier-1 | Chelsea signed Moises Caicedo from Brighton for £115M in January 2024, contract until 2031 |
| `tier-2/injury-report.md` | tier-2 | Kevin De Bruyne suffered a hamstring injury during training on 10 Feb 2025, out for six weeks |
| `tier-2/contract.md` | tier-2 | Erling Haaland extended his Manchester City contract until 2034 (announced May 2024) |

Deliberately small and hand-written: the point is a corpus whose ground truth
a human can verify by reading it in thirty seconds, not a realistic-scale
dataset. Load/scale testing is a separate, not-yet-started Phase 8 item.

## The questions

`eval/questions.json` holds 6 hand-curated questions: 4 answerable from the
corpus above, and 2 deliberately not, to test whether the system correctly
declines instead of guessing.

| ID | Query | Tier | Expected answerable? | Expected source |
|---|---|---|---|---|
| `q1-derby-score` | Who won the north London derby between Arsenal and Tottenham, and what was the score? | tier-1 | Yes | `tier-1/derby.md` |
| `q2-caicedo-fee` | How much did Chelsea pay to sign Moises Caicedo from Brighton? | tier-1 | Yes | `tier-1/transfer.md` |
| `q3-debruyne-injury` | What injury did Kevin De Bruyne suffer, and when? | tier-2 | Yes | `tier-2/injury-report.md` |
| `q4-haaland-contract` | Until what year did Erling Haaland extend his contract with Manchester City? | tier-2 | Yes | `tier-2/contract.md` |
| `q5-unanswerable-world-cup` | Who won the FIFA World Cup in 1998? | tier-1 | No — not in the corpus | — |
| `q6-unanswerable-messi-transfer` | What was Lionel Messi's transfer fee when he joined Inter Miami? | tier-2 | No — not in the corpus | — |

## What each metric means

**`retrieval_precision`** — of the questions that *should* be answerable,
what fraction did retrieval actually surface the right source document for?
Computed deterministically: for each answerable question, checks whether any
of the citations the pipeline returned matches that question's hand-curated
`expected_source_paths`. This measures the retrieval half of RAG in
isolation — it can be perfect even if the generated wording is later judged
unfaithful, because that's a separate failure mode measured separately.

**`faithfulness_rate`** — of the questions the system actually answered
*and* was expected to be able to answer, what fraction of the generated
answers are fully supported by the cited source text? This is the one
genuinely subjective dimension in the whole evaluation, so it's the only
metric that goes through an LLM judge rather than a deterministic check
(`evaluation/judge.py::check_faithfulness()`, reusing the same CLEAN-vs-
flagged judging pattern the three production security judges already use,
judged by a separate model — `qwen2.5:14b-instruct` — running at
`temperature=0.0`, distinct from both the generation model and the judges
guarding the live request path). A "CLEAN" verdict means every claim in the
answer traces back to what was actually cited; a flagged verdict means the
answer said something the cited source doesn't support.

**`hallucination_rate`** — of every question that was actually scored, what
fraction did the system either fabricate an answer to (when it should have
declined) or answer unfaithfully (when it was faithfully answerable)? This
is the headline safety number: a system that hallucinates gives a user
false confidence in something untrue. It has two distinct triggers that are
never conflated: answering at all on a question that wasn't expected to be
answerable (fabrication where nothing should have been said), or answering
an expected-answerable question in a way the faithfulness judge flags. A
question that *should* have been answerable but got the canonical
"I don't know" fallback instead is neither of these — that's counted as a
retrieval miss, a different, less alarming failure mode.

**`errored_count`** — how many questions the pipeline itself failed to
score at all (a judge-model timeout/OOM, an Ollama connection failure —
this machine's own hardware has a documented history of exactly that under
load). An errored question is excluded from every *other correctness*
metric's denominator (`retrieval_precision`, `faithfulness_rate`,
`hallucination_rate`) rather than being counted as a placeholder "not
hallucinated" — a question the system never got to answer is not evidence
the system behaved correctly on it. Timing is the one exception: an errored
question's `duration_seconds` still counts toward `average_duration_seconds`
below, since "how long before it failed" is a real measurement even when
the answer isn't.

**`duration_seconds` (per question) / `average_duration_seconds`
(aggregate)** — real wall-clock time for one question's full round trip:
retrieval, generation, and (when it runs) the faithfulness judge call. Not a
proxy or an estimate — measured with `time.monotonic()` around the actual
pipeline call. Unlike the three correctness metrics above,
`average_duration_seconds` is computed over *every* question, including
errored ones, on the reasoning that a slow failure is still a real data
point about system health, unlike the correctness metrics, which an error
genuinely has nothing to say about.

## Reading a report

Every run writes a fresh, timestamped `eval/results/eval-<timestamp>.json`
(gitignored — it's run output, not a fixture, so it's never committed and
will look different for every checkout and every run). This section is a
worked example of how to read one, not a live claim about the system's
current behavior — treat the numbers below as illustrative, not current.

Example report, run against the real pipeline (`mistral` generation,
`qwen2.5:14b-instruct` faithfulness judging, real embedded Qdrant, real
Ollama — nothing mocked):

| Metric | Value |
|---|---|
| `retrieval_precision` | 1.0 (4/4 answerable questions) |
| `faithfulness_rate` | 0.75 (3/4 judged answers) |
| `hallucination_rate` | 0.167 (1/6 questions) |
| `errored_count` | 0 |
| `average_duration_seconds` | 76.7s |

### Per-question breakdown

| ID | Answer | Retrieval hit | Faithfulness | Hallucinated | Duration |
|---|---|---|---|---|---|
| `q1-derby-score` | "Arsenal won the North London Derby with a score of 3-1 [1]." | ✅ | CLEAN | No | 80.6s |
| `q2-caicedo-fee` | "[1] £115 million" | ✅ | CLEAN | No | 100.3s |
| `q3-debruyne-injury` | "Kevin De Bruyne suffered a hamstring injury on 10 February 2025 [1]." | ✅ | UNSUPPORTED | Yes | 88.5s |
| `q4-haaland-contract` | "[1] Erling Haaland extended his contract with Manchester City until 2034." | ✅ | CLEAN | No | 101.4s |
| `q5-unanswerable-world-cup` | "I do not know the answer based on indexed documents." | n/a (correctly declined) | n/a | No | 52.3s |
| `q6-unanswerable-messi-transfer` | "I do not know the answer based on indexed documents." | n/a (correctly declined) | n/a | No | 36.8s |

Reading it: retrieval went 4/4, both deliberately-unanswerable questions
were correctly declined (zero fabrication), and one answerable question
(`q3`) was flagged unfaithful by the judge — on inspection, a fair
paraphrase of its source that omitted one qualifying detail ("during
training"), which looks like judge calibration rather than a real defect
(see `PROJECT_TRACKER.md`'s Phase 8 entry for the full note). Answerable
questions (`q1`–`q4`) ran noticeably slower than declined ones (`q5`, `q6`)
in this sample, since only an answerable question triggers a faithfulness
judge call on top of generation — on this development machine (a 4GB-VRAM
laptop GPU running both models mostly via CPU offload), that's the
dominant cost, not anything architectural in the pipeline itself. Re-run
`python -m agentic_rag.evaluation.runner` for current numbers before
drawing any conclusion from this example.

## What to look for when reading a report

These are the questions a report should be read against — not conclusions
about the system's current state (see the caveat above: numbers go stale
the moment a new run happens), but the standard this evaluation exists to
hold every run to:

**Does retrieval find the right source for every answerable question?**
`retrieval_precision` below 1.0 on a corpus this small and unambiguous is
worth investigating immediately — it means hybrid (dense+sparse) search
with reranking missed a document a human would find trivially.

**Does the system fabricate on questions it shouldn't answer?** This is
the single most important number in the report, for a system whose stated
design philosophy is "never invents facts." Any non-zero fabrication (an
unanswerable question, `hallucinated: true`, `error: null`) is a regression
worth blocking on, not averaging away.

**Is a faithfulness flag a real defect or judge calibration?** Read the
flagged answer against its cited source by hand before assuming either. A
single flagged paraphrase that's merely less detailed than its source (not
actually false) is a data point about the judge's strictness, not
necessarily a system defect — worth tracking across runs, not
over-indexing on from one example (see `PROJECT_TRACKER.md`'s Phase 8 entry
for a worked example of this distinction from a real run).

**Is response time representative of the environment it was measured in?**
On constrained local hardware (this project's own development machine is a
4GB-VRAM laptop GPU running generation and a 14B-parameter judge model
mostly via CPU offload), `average_duration_seconds` reflects that hardware,
not the pipeline's architecture. Don't treat a number measured here as a
production latency budget without re-measuring on the target environment.

**Is the sample size big enough to trust?** 6 questions over 4 documents is
enough to catch a broken pipeline, not enough to make a statistically
confident claim about faithfulness or hallucination rates in general.
Growing the question set (more paraphrases, more edge cases, more
documents) is the natural next step for this evaluation to become more than
a smoke test — tracked as future work, not started here.
