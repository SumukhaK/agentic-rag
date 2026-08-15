# Load test

This directory holds the Phase 8 load test: does this project's pipeline
actually handle `docs/REQUIREMENTS.md` §2's real target — **at least 10,000
documents, averaging ~50 pages each (≈500,000 pages)** — not the
theoretical, 10-document-calibration extrapolation in README.md's "Scaling
to 150,000 Documents" section, but a real run against the real pipeline.

Full design rationale and self-review history live in
[`PROJECT_TRACKER.md`](../PROJECT_TRACKER.md), Phase 8. This file is the
plain-language reference: how to actually run it, what the two-phase
approach measures, and what to do if it crashes partway through.

**This code has been live-verified only at small scale (a few dozen
documents), not at the real 10,000-document target.** A run at target scale
takes on the order of a day of continuous indexing on typical development
hardware (see "Expected duration" below) — treat the first real run as the
actual validation of this code, not something already proven.

## How to run it

Two steps, in order:

```bash
# 1. Generate the synthetic corpus (deterministic - same --seed always
#    regenerates identical documents, so nothing here needs to be
#    committed, even though the script that produces it is).
python -m agentic_rag.loadtest.corpus_generator

# 2. Run the load test: batch-indexes the generated corpus through the
#    real pipeline, then measures query latency against the fully-loaded
#    index.
python -m agentic_rag.loadtest.runner
```

Both default to `docs/REQUIREMENTS.md`'s real target (10,000 documents,
~50 pages each). `corpus_generator.py` also accepts `--document-count`,
`--pages-per-document`, `--seed`, and `--output` for a smaller smoke-test
run - e.g. `python -m agentic_rag.loadtest.corpus_generator
--document-count 20 --output /tmp/loadtest-smoke` to validate the harness
itself before committing to the real run.

## What it measures, and why two phases

**Ingestion phase.** The generated corpus is drip-fed into a dedicated
watched folder in `LOADTEST_BATCH_SIZE`-sized batches (default 200),
indexed through `run_sync_cycle()` - the exact same code path the
background sync job and `POST /query` share in production, not a
separate benchmark harness. One batch, one `run_sync_cycle()` call, one
checkpoint saved immediately after.

**Query-latency phase.** README.md's 150k theoretical analysis only ever
measured *ingestion* throughput. `docs/REQUIREMENTS.md` §2's actual
requirement is "fast and reliable" for *querying* - so once the full
corpus is indexed, a handful of representative queries are answered
through `answer_with_cache()` (the same function `POST /query` calls)
against the now fully-loaded index, and their latency is recorded. This
is the number that says whether the system is actually usable at target
scale, not just whether the data fits.

Everything runs against a dedicated Qdrant collection/storage path
(`LOADTEST_QDRANT_*`), never the app's real ones or the eval corpus's -
same isolation reasoning `eval/README.md`'s own dedicated collection
already documents.

## Expected duration and resource use

Per README.md's own calibration run (≈10.6s/document at this project's
typical development hardware), the full 10,000-document ingestion phase
is expected to take **on the order of a day of continuous indexing** -
long enough that it should be started as a detached/background process,
not run in a terminal you need to keep open. Expect roughly **9-10GB of
Qdrant storage** and a **few GB of resident HNSW index in RAM** at target
scale (both scaled down 15x from the 150k theoretical analysis's own
extrapolated figures).

## Resuming after a crash

The load test is designed to lose at most one batch's worth of progress
(~35 minutes at the default batch size), not the whole run:

- Every batch is checkpointed via `save_snapshot()` immediately after its
  `run_sync_cycle()` call returns - the same atomic-write primitive
  `run_sync_loop()` already uses in production.
- On restart, `python -m agentic_rag.loadtest.runner` recomputes which
  staged documents aren't yet present in the loadtest watched folder as
  the remaining work - no separate progress file to go stale or get out
  of sync with reality.
- Simply re-running the same command after a crash (for whatever reason -
  an OOM, a machine restart, a closed terminal) picks up exactly where it
  left off.

## Results

Every run writes a fresh, timestamped
`loadtest/results/loadtest-<timestamp>.json` (gitignored - run output, not
a fixture) with the ingestion totals, per-batch timing, and the
query-latency phase's measurements. **No real 10,000-document run has been
completed yet** - this section will be updated with real numbers, and
compared against the 150k theoretical analysis's predictions, once one
has.
