from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

# Matches README.md's own "Scaling to 150,000 Documents" calibration
# convention (≈150,040 chars measured for a ~50-page document, ≈3,000
# chars/page) - keeping the same chars-per-page basis makes a real run's
# measured numbers directly comparable to that theoretical extrapolation.
CHARS_PER_PAGE = 3000

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


def _team_name(rng: random.Random) -> str:
    return f"{rng.choice(_CITY_NAMES)} {rng.choice(_CLUB_SUFFIXES)}"


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


def _tier_for_index(index: int, document_count: int, access_tiers: Sequence[str]) -> str:
    """Which `access_tiers` subfolder document `index` belongs in, split
    as evenly as possible across the tiers in order (`employee` first,
    then `manager`, ...) - matches the real corpus's own tier-per-subfolder
    convention (`eval/README.md`'s corpus layout).
    """
    tier_size = -(-document_count // len(access_tiers))  # ceil division
    tier_position = min(index // tier_size, len(access_tiers) - 1)
    return access_tiers[tier_position]


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


if __name__ == "__main__":
    main()
