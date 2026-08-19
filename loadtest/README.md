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

**Unlike the eval corpus, the loadtest collection is never deleted and
recreated between runs** - it persists deliberately, since that's what
makes crash-resumption possible (see below). This means regenerating
the staged corpus with different `--document-count`/`--pages-per-
document`/`--seed` values does **not** give you a clean slate on its
own: old documents from a prior run stay indexed unless you clear
`LOADTEST_WATCHED_FOLDER_PATH`, `LOADTEST_QDRANT_STORAGE_PATH`, and
`LOADTEST_SYNC_SNAPSHOT_PATH` first. Clear all three before generating
a differently-parameterized corpus, or the "fully-loaded index" the
query-latency phase measures against will silently be a mix of two
different corpora.

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
query-latency phase's measurements - but only if `run_load_test()` returns
normally. The one real run attempted so far never reached that point (see
below), so this section is a manual write-up from the structured batch
logs and direct on-disk inspection, not a report file.

### What was attempted

A real run against the full `docs/REQUIREMENTS.md` §2 target - 10,000
documents, ~50 pages each, the default `loadtest_batch_size=200` - was
started on 2026-08-16 and worked on, on and off, through 2026-08-19, on
this project's own development machine (GTX 1650 Ti, 4GB VRAM, 16GB
system RAM). **It did not complete.** It was stopped at **6,000 / 10,000
documents (60%)** after repeatedly hitting a real memory-scaling wall this
hardware could not sustain past that point - not a code bug, and not
something a restart could fix, since it recurred after every restart
within a few hours.

### Measured results (0 - 6,000 documents)

| Metric | Measured value |
|---|---|
| Documents indexed | 6,000 / 10,000 (60%) |
| Total chunks indexed | 476,439 (79.4/doc - matches the 150k analysis's own 79/doc calibration almost exactly) |
| Qdrant storage | 5.5 GB (≈0.94 MB/doc - close to the 150k analysis's 0.93 MB/doc calibration) |
| Indexing failures | 1 (`tier-1/doc_02197.md`, isolated - 5,999/6,000 succeeded cleanly) |
| Total logged indexing time (sum of completed batches) | ≈146,613s (≈40.7 hours) |
| Total real wall-clock time (first launch to final stop) | ≈72.7 hours (≈3.0 days) - the gap between this and the row above is the two multi-hour unexplained stalls/crashes below, not active work |

### The rate degraded by ~3x partway through, then the process started dying

Per-batch timing split cleanly into two regimes, with no gradual ramp
between them:

| Phase | Batches | Docs | Avg time/batch | Avg time/doc |
|---|---|---|---|---|
| "Healthy" (0-23) | 24 | 4,800 | ≈58.3 min | ≈17.5s |
| "Degraded" (24-29) | 6 | 1,200 | ≈174.1 min (**≈3.0x slower**) | ≈52.2s |

The healthy-phase per-document rate (≈17.5s) already ran somewhat slower
than the 150k analysis's own 10-document calibration (≈10.6s/doc) - a real
hardware/timing variance the small calibration sample couldn't have
caught. The degraded phase is the more important finding: `nvidia-smi`
showed no thermal throttling (60-62°C, GPU idle between calls, low VRAM
use) when checked during the slow batches, ruling out a GPU heat problem.
What was actually happening: **system RAM was being exhausted.** Free
system memory (16GB total) was repeatedly observed down to
**~1GB free**, with the load-test process's own working set climbing past
6.6GB and still growing. This matches, and empirically confirms, exactly
what the 150k theoretical analysis flagged as an unquantified risk
("HNSW insert cost is known to grow with graph size... neither effect is
quantified here") - except it showed up at **6,000 documents**, well
short of even this project's own real 10,000-document target, not just in
a 15x-larger hypothetical scenario.

### The process required three restarts, and never fully recovered

- After batch 29 (6,000 docs), the process died with **no error, no
  traceback, and no recorded machine reboot** - consistent with a silent
  OS-level OOM kill under the RAM pressure above.
- Resumed via `python -m agentic_rag.loadtest.runner` (the crash-recovery
  design worked exactly as built and tested: the stranded batch was picked
  up automatically, no data lost) - but the new process hit the same wall
  and produced **zero further progress across the next several hours**,
  with system RAM back down to ~1GB free and command-level system
  operations (even a plain `ls`) visibly slowed by the resulting paging.
- Killed and resumed a second time; the same pattern recurred within
  hours.
- At that point the run was stopped deliberately rather than continuing
  to fight the same wall - 60% of the real target, with a clear, repeated,
  measured cause, is a more informative result than an indefinite series
  of manual restarts chasing a hardware limit no restart can fix.

### What this means

**This is a real, negative result, not a failed measurement.** The 150k
theoretical analysis (above) predicted this project's current
architecture - a single process, embedded/local Qdrant holding its whole
HNSW graph resident in memory, no sharding or on-disk vector storage -
would not scale gracefully; this run demonstrates that limit is reached
well before even the real 10,000-document target on hardware in this
project's own class (4GB VRAM, 16GB RAM). Extrapolating the *degraded*
rate (≈52.2s/doc) for the remaining 4,000 documents gives ≈58 more hours
of indexing alone - and that estimate is almost certainly optimistic,
since it assumes the process could run that whole stretch without another
OOM-driven stall, which it demonstrably could not do even once already.

The query-latency phase (measuring `POST /query` latency against the
fully-loaded index) never ran, since `run_load_test()` never reached it -
that measurement remains genuinely unmeasured, not just unfavorable.

**What would actually fix this**, per the 150k analysis's own "where the
current architecture breaks down" section: moving off embedded/in-memory
Qdrant (a real Qdrant server with on-disk vector storage, or sharding)
before attempting this scale again - not a load-test script change. That
remains its own architectural decision, requiring its own ADR, not
something to retry as-is expecting a different outcome.
