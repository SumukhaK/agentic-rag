# `loadtest/corpus_generator.py`

**Purpose:** This file manufactures a large, fake-but-realistic set of football "match report" documents that the system's load test can ingest and index, instead of requiring a real 10,000-document, multi-gigabyte corpus to be checked into the repository. It is deliberately built so that running it twice with the same inputs produces byte-identical output (so the test is repeatable) and so that no two generated paragraphs anywhere in the entire corpus are ever identical (so the embedding cache used elsewhere in the system can't accidentally treat the load test as smaller/easier than it really is by recognizing "repeated" text). The documents it writes are split across access-tier subfolders (`employee`, `manager`, `director`) so the rest of the load test can exercise the platform's tier-based access control at scale.

## Line-by-line walkthrough

### Lines 1-6 — Imports
```python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence
```
- `from __future__ import annotations` — makes all type hints in the file lazily evaluated as strings rather than executed immediately. This lets the code use modern type hint syntax (like `tuple[str, ...]`) safely across Python versions without extra imports.
- `import argparse` — used later to build the command-line interface for running this file directly (`--document-count`, `--pages-per-document`, etc.).
- `import random` — used to deterministically generate pseudo-random football content (team names, scores, tactics) from a seed.
- `from pathlib import Path` — used throughout for filesystem paths (the output directory, per-tier subfolders, individual document files) instead of raw strings.
- `from typing import Sequence` — used to type-hint the `access_tiers` parameter as "any ordered, indexable collection of strings" without committing to a specific container type like `list` or `tuple`.

### Lines 8-12 — `CHARS_PER_PAGE` constant
```python
# Matches README.md's own "Scaling to 150,000 Documents" calibration
# convention (≈150,040 chars measured for a ~50-page document, ≈3,000
# chars/page) - keeping the same chars-per-page basis makes a real run's
# measured numbers directly comparable to that theoretical extrapolation.
CHARS_PER_PAGE = 3000
```
- The comment explains that `3000` isn't an arbitrary number — it mirrors a calibration figure already used in the project's README when estimating how the system would scale to 150,000 documents. Reusing the same chars-per-page constant means results measured by an actual load test run can be directly compared against that earlier back-of-envelope theoretical estimate, rather than the two using different, incompatible units.
- `CHARS_PER_PAGE = 3000` — defines the constant itself: each simulated "page" of a document is treated as roughly 3000 characters of generated text.

### Lines 14-24 — `DEFAULT_ACCESS_TIERS` constant
```python
# The load test's own, dedicated access-tier list - deliberately NOT
# Settings.access_tiers. The generated corpus's folder layout and every
# loadtest component that needs to agree on tier names (this generator,
# runner.py's batch_settings override, and its representative queries'
# known_tiers) import this single constant, so the load test stays fully
# self-contained regardless of what ACCESS_TIERS happens to be configured
# to for the real app - a customized ACCESS_TIERS env var would otherwise
# make every generated employee/manager/director document fail tagging
# (corpus folders wouldn't match the configured tiers), silently zeroing
# out the whole ingestion phase.
DEFAULT_ACCESS_TIERS: tuple[str, ...] = ("employee", "manager", "director")
```
- The comment explains an important design decision: this constant is intentionally separate from the main application's configurable `Settings.access_tiers`. If the load test instead read the tier list from environment-configurable settings, then someone customizing the real app's access tiers (e.g. renaming `employee` to something else) would silently break the load test — the generated folders would no longer match the tiers the rest of the pipeline expects, and every document would fail to be tagged, causing ingestion to quietly index nothing. By hardcoding a dedicated constant that every load-test file imports, the load test is self-contained and immune to that kind of configuration drift.
- `DEFAULT_ACCESS_TIERS: tuple[str, ...] = ("employee", "manager", "director")` — defines the three tiers used to organize the generated corpus into subfolders, in a fixed, ordered tuple.

### Lines 26-59 — Word pools for generated text
```python
_CITY_NAMES = [
    "Ashford", "Brackenfield", "Corvale", "Dunmoor", "Elmsworth", "Fenwick",
    "Gladstone", "Harrow Vale", "Ironbridge", "Kestrel Bay", "Lansdowne",
    "Mossgate", "Norwood", "Oakhaven", "Pemberton", "Queensferry",
    "Ravensmoor", "Stonefield", "Thornbury", "Uxmoor",
]
_CLUB_SUFFIXES = [
    "United", "City", "Rovers", "Athletic", "Town", "Wanderers", "Albion", "Rangers",
]
_COMPETITIONS = [
    "Premier Division", "Continental Cup", "National Trophy",
    "Regional Championship", "First Division", "Super League",
]
_POSITIONS = [
    "forward", "midfielder", "central defender", "goalkeeper", "winger", "full-back",
]
_TACTICS = [
    "a high press", "a low defensive block", "patient possession-based build-up",
    "direct counter-attacking transitions", "overlapping full-backs",
    "inverted wingers cutting inside", "a back three", "a double pivot in midfield",
]
_OPENERS = [
    "In a closely contested fixture,", "Following a slow start,",
    "In front of a vocal home crowd,", "Under difficult weather conditions,",
    "In a match shaped by two red cards,", "Despite an early setback,",
]
_ANALYSIS = [
    "The manager praised the squad's discipline after the final whistle.",
    "Statistical models had rated this fixture as evenly balanced.",
    "Injuries in the back line forced a late tactical reshuffle.",
    "Set-piece delivery proved decisive in the closing stages.",
    "The result leaves the table finely poised heading into the final stretch.",
    "Pressing intensity dropped sharply after the hour mark.",
]
```
- These module-level (leading underscore means "private to this file") lists are the raw vocabulary the text generator draws from. Each list covers one "slot" in a templated sentence:
  - `_CITY_NAMES` — fictional city names used to build team names (e.g. "Ashford").
  - `_CLUB_SUFFIXES` — football club naming suffixes (e.g. "United", "City") combined with a city name to form a full club name (e.g. "Ashford United").
  - `_COMPETITIONS` — names of fictional competitions a match report might belong to.
  - `_POSITIONS` — player positions mentioned as being "influential" in a match.
  - `_TACTICS` — tactical approaches a team might be described as using.
  - `_OPENERS` — sentence-opening phrases to vary how each paragraph begins.
  - `_ANALYSIS` — closing analytical sentences summarizing an aspect of the match.
- Combining values from these small pools by random choice produces plausible-sounding football journalism text without needing any real external data source.

### Lines 62-63 — `_team_name`
```python
def _team_name(rng: random.Random) -> str:
    return f"{rng.choice(_CITY_NAMES)} {rng.choice(_CLUB_SUFFIXES)}"
```
- Defines a helper that builds one team name by randomly picking a city and a club suffix and joining them with a space (e.g. "Kestrel Bay Rovers"). Takes an explicit `random.Random` instance (`rng`) rather than using the global `random` module, so the caller controls exactly which random number generator (and therefore which seed) drives the choice — this is what makes output reproducible.

### Lines 66-89 — `_generate_paragraph`
```python
def _generate_paragraph(rng: random.Random, doc_index: int, paragraph_index: int) -> str:
    """One football-domain-styled paragraph (~250-350 chars).

    Ends with an explicit `(ref: {doc_index}-{paragraph_index})` tag -
    `doc_index` is unique per document across the whole corpus and
    `paragraph_index` is unique within a document, so this tag alone
    already guarantees no two paragraphs in the entire corpus are
    byte-identical, regardless of how small the word pools above are.
    This is what makes `_generate_document_text()`'s uniqueness
    guarantee hold structurally rather than just statistically - the
    exact property README.md's calibration run got wrong the first time
    (a small, cycling paragraph pool let `EmbeddingCache` treat most
    chunks as repeats).
    """
    team_a = _team_name(rng)
    team_b = _team_name(rng)
    score_a, score_b = rng.randint(0, 5), rng.randint(0, 5)
    minute = rng.randint(1, 90)
    return (
        f"{rng.choice(_OPENERS)} {team_a} defeated {team_b} {score_a}-{score_b} "
        f"in the {rng.choice(_COMPETITIONS)}. The {rng.choice(_POSITIONS)} was "
        f"influential, deploying {rng.choice(_TACTICS)} from the {minute}th minute "
        f"onward. {rng.choice(_ANALYSIS)} (ref: {doc_index}-{paragraph_index})"
    )
```
- The docstring explains the key design decision in this function: every generated paragraph ends with a tag like `(ref: 42-7)` combining the document's index and the paragraph's position within that document. Because `doc_index` is unique across the whole corpus and `paragraph_index` is unique within a document, this tag alone guarantees every paragraph in the entire corpus is textually unique — even if the word pools above are small enough that two paragraphs might otherwise randomly end up wording things identically. The docstring notes this fixes a real earlier mistake: without a guaranteed-unique tag, a small cycling pool of words could cause many paragraphs to coincidentally repeat, and the system's `EmbeddingCache` (which skips re-embedding text it's seen before) would then treat most of the "large" corpus as cached duplicates — silently making the load test far less realistic than intended.
- `def _generate_paragraph(rng, doc_index, paragraph_index) -> str:` — takes the shared random generator plus the two indices needed to build the uniqueness tag.
- `team_a = _team_name(rng)` / `team_b = _team_name(rng)` — picks two (possibly coincidentally identical, though this doesn't affect uniqueness since the tag is what guarantees it) team names for this paragraph's fictional match.
- `score_a, score_b = rng.randint(0, 5), rng.randint(0, 5)` — picks a plausible football scoreline (0 to 5 goals for each side).
- `minute = rng.randint(1, 90)` — picks a random minute in a standard 90-minute match, used later in the sentence.
- The `return (...)` f-string assembles the full paragraph: an opener, the two team names and score, the competition, an "influential" player position, a tactic phrase and the minute it was deployed from, a closing analysis sentence, and finally the unique `(ref: ...)` tag.

### Lines 92-112 — `_generate_document_text`
```python
def _generate_document_text(index: int, seed: int, *, pages: int) -> str:
    """Deterministic, unique synthetic document text for `index` under `seed`.

    `random.Random(seed * 1_000_003 + index)` - a plain int derived from
    both, not a tuple (which `random.Random()` doesn't reliably accept as
    a seed) - makes generation fully reproducible: re-running the
    generator with the same `seed` regenerates byte-identical output
    without needing to commit the ~1.5GB a real 10,000-document corpus
    would take.
    """
    rng = random.Random(seed * 1_000_003 + index)
    target_chars = pages * CHARS_PER_PAGE
    paragraphs: list[str] = []
    total_chars = 0
    paragraph_index = 0
    while total_chars < target_chars:
        paragraph = _generate_paragraph(rng, index, paragraph_index)
        paragraphs.append(paragraph)
        total_chars += len(paragraph) + 2
        paragraph_index += 1
    return "\n\n".join(paragraphs)
```
- The docstring explains why the random seed is computed as `seed * 1_000_003 + index` rather than, say, seeding with a `(seed, index)` tuple: Python's `random.Random()` doesn't reliably accept a tuple as a seed, so the two values are combined into a single plain integer instead (multiplying by a large prime-like number and adding `index` avoids collisions between different `(seed, index)` combinations landing on the same effective seed). The practical benefit: given the same `seed`, calling this function again for the same `index` always produces exactly the same text. That means the full ~1.5GB corpus never needs to be stored in the repository — it can be regenerated on demand, byte-for-byte identical, from just the seed and document count.
- `rng = random.Random(seed * 1_000_003 + index)` — creates a per-document random generator seeded uniquely and deterministically, per the reasoning above.
- `target_chars = pages * CHARS_PER_PAGE` — computes how many characters this document should contain in total, based on the requested page count and the `CHARS_PER_PAGE` constant.
- `paragraphs: list[str] = []` — accumulator list for the paragraphs generated so far.
- `total_chars = 0` — running count of characters generated so far, compared against `target_chars` to know when to stop.
- `paragraph_index = 0` — tracks which paragraph number within this document is being generated next (used in the uniqueness tag).
- `while total_chars < target_chars:` — keeps generating paragraphs until the document has reached (or slightly exceeded) its target length.
  - `paragraph = _generate_paragraph(rng, index, paragraph_index)` — generates the next paragraph using the shared per-document RNG.
  - `paragraphs.append(paragraph)` — adds it to the list.
  - `total_chars += len(paragraph) + 2` — updates the running character count; the `+ 2` accounts for the two newline characters (`\n\n`) that will separate this paragraph from the next when joined.
  - `paragraph_index += 1` — advances to the next paragraph number.
- `return "\n\n".join(paragraphs)` — joins all generated paragraphs with a blank line between them (standard markdown paragraph separation) and returns the full document body text.

### Lines 115-123 — `_tier_for_index`
```python
def _tier_for_index(index: int, document_count: int, access_tiers: Sequence[str]) -> str:
    """Which `access_tiers` subfolder document `index` belongs in, split
    as evenly as possible across the tiers in order (`employee` first,
    then `manager`, ...) - matches the real corpus's own tier-per-subfolder
    convention (`eval/README.md`'s corpus layout).
    """
    tier_size = -(-document_count // len(access_tiers))  # ceil division
    tier_position = min(index // tier_size, len(access_tiers) - 1)
    return access_tiers[tier_position]
```
- The docstring explains the goal: split the corpus as evenly as possible across the given tiers, assigning documents to tiers in order (all of tier 1's share first, then tier 2's, and so on) — matching the folder-per-tier layout convention already used elsewhere in the project's evaluation corpus.
- `tier_size = -(-document_count // len(access_tiers))  # ceil division` — computes how many documents belong in each tier, rounding up. The double-negative trick (`-(-a // b)`) is a common Python idiom for ceiling integer division without importing `math.ceil`, since regular `//` rounds down. Rounding up ensures every document gets assigned to some tier even if `document_count` doesn't divide evenly.
- `tier_position = min(index // tier_size, len(access_tiers) - 1)` — determines which tier index this document falls into by integer-dividing its position by the tier size, then clamps the result to never exceed the last valid tier index. The clamp (`min(..., len(access_tiers) - 1)`) protects against an off-by-one edge case: because `tier_size` is rounded up, the very last few documents could otherwise compute a `tier_position` one past the end of the list.
- `return access_tiers[tier_position]` — looks up and returns the actual tier name (e.g. `"employee"`) for that computed position.

### Lines 126-148 — `generate_corpus`
```python
def generate_corpus(
    staging_path: Path,
    *,
    document_count: int = 10_000,
    pages_per_document: int = 50,
    access_tiers: Sequence[str] = DEFAULT_ACCESS_TIERS,
    seed: int = 0,
) -> None:
    """Write `document_count` synthetic, football-domain-styled markdown
    documents to `staging_path`, distributed across `access_tiers`
    subfolders. Written to a staging directory, not directly to a watched
    folder - `loadtest/runner.py` drip-feeds this staged corpus into the
    watched folder in batches, so a single ~30-hour `run_sync_cycle()`
    call is never asked to process the entire corpus in one uncheckpointed
    pass.
    """
    for index in range(document_count):
        tier = _tier_for_index(index, document_count, access_tiers)
        text = _generate_document_text(index, seed, pages=pages_per_document)
        tier_dir = staging_path / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tier_dir / f"doc_{index:05d}.md"
        doc_path.write_text(f"# Match Report {index:05d}\n\n{text}\n", encoding="utf-8")
```
- The function signature: `staging_path` is the only required, positional argument (where to write generated files); everything after the bare `*` must be passed by keyword. Defaults: `document_count=10_000` documents, `pages_per_document=50` pages each, `access_tiers=DEFAULT_ACCESS_TIERS`, and `seed=0` for reproducibility.
- The docstring explains an important architectural point: this function writes to a "staging" directory rather than directly to the folder the ingestion pipeline actively watches. The reason is in `runner.py`: it feeds this staged corpus into the watched folder gradually, in checkpointed (progress-saving) batches, because processing the entire ~10,000-document corpus in one uninterrupted pass could take roughly 30 hours — far too long to safely run as a single all-or-nothing operation without the ability to resume after a crash.
- `for index in range(document_count):` — loops once per document to be generated, `index` from `0` to `document_count - 1`.
  - `tier = _tier_for_index(index, document_count, access_tiers)` — determines which access-tier subfolder this document belongs in.
  - `text = _generate_document_text(index, seed, pages=pages_per_document)` — generates this document's deterministic, unique body text.
  - `tier_dir = staging_path / tier` — computes the path to this tier's subfolder under the staging directory.
  - `tier_dir.mkdir(parents=True, exist_ok=True)` — creates that subfolder (and any missing parent directories) if it doesn't already exist; `exist_ok=True` means it's not an error if it's already there (important since this runs once per document but the same tier folder is reused many times).
  - `doc_path = tier_dir / f"doc_{index:05d}.md"` — builds the filename, zero-padding the index to 5 digits (e.g. `doc_00042.md`) so files sort in a predictable, consistent order.
  - `doc_path.write_text(f"# Match Report {index:05d}\n\n{text}\n", encoding="utf-8")` — writes the file: a markdown level-1 heading naming the report, a blank line, the generated body text, and a trailing newline, encoded as UTF-8.

### Lines 151-171 — `main` (CLI entry point)
```python
def main() -> None:
    """CLI entry point: `python -m agentic_rag.loadtest.corpus_generator`."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic, football-domain-styled corpus for the "
            "Phase 8 load test (docs/REQUIREMENTS.md §2)."
        )
    )
    parser.add_argument("--document-count", type=int, default=10_000)
    parser.add_argument("--pages-per-document", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("./loadtest/corpus_staging"))
    args = parser.parse_args()

    generate_corpus(
        args.output,
        document_count=args.document_count,
        pages_per_document=args.pages_per_document,
        seed=args.seed,
    )
    print(f"generated {args.document_count} documents to {args.output}")
```
- The docstring notes how this function is meant to be invoked: as a module run directly from the command line (`python -m agentic_rag.loadtest.corpus_generator`).
- `parser = argparse.ArgumentParser(description=...)` — creates the command-line argument parser, with a human-readable description (shown in `--help` output) referencing where this load test phase is specified in the project's requirements document.
- `parser.add_argument("--document-count", type=int, default=10_000)` — lets the caller override how many documents to generate; defaults to 10,000.
- `parser.add_argument("--pages-per-document", type=int, default=50)` — lets the caller override the page count per document; defaults to 50.
- `parser.add_argument("--seed", type=int, default=0)` — lets the caller override the random seed for reproducibility; defaults to 0.
- `parser.add_argument("--output", type=Path, default=Path("./loadtest/corpus_staging"))` — lets the caller override the output/staging directory; defaults to a relative `./loadtest/corpus_staging` folder, parsed directly as a `Path` object.
- `args = parser.parse_args()` — parses the actual command-line arguments the script was invoked with.
- `generate_corpus(args.output, document_count=..., pages_per_document=..., seed=...)` — calls the main generation function with the parsed values. Note `access_tiers` isn't passed here, so it falls back to `generate_corpus`'s own default (`DEFAULT_ACCESS_TIERS`) — the CLI doesn't expose a way to override the tier list, consistent with the earlier design decision that the load test's tiers are meant to stay fixed.
- `print(f"generated {args.document_count} documents to {args.output}")` — prints a simple confirmation message once generation completes.

### Lines 174-175 — Script entry guard
```python
if __name__ == "__main__":
    main()
```
- Standard Python idiom: only calls `main()` when this file is executed directly (e.g. via `python -m agentic_rag.loadtest.corpus_generator`), not when it's imported as a module by other code (such as `runner.py`, which imports `DEFAULT_ACCESS_TIERS` from this file without wanting the CLI to run).
