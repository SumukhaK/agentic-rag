from pathlib import Path

from agentic_rag.loadtest.corpus_generator import (
    CHARS_PER_PAGE,
    _generate_document_text,
    _tier_for_index,
    generate_corpus,
)
from tests.access_tiers import ACCESS_TIERS, TIER_DIRECTOR, TIER_EMPLOYEE, TIER_MANAGER


def test_generate_document_text_length_is_approximately_pages_times_chars_per_page():
    text = _generate_document_text(0, seed=0, pages=50)

    target = 50 * CHARS_PER_PAGE
    # A paragraph-boundary loop can't land on the target exactly - it
    # stops once the target is reached, so the result is always >= target
    # and, since paragraphs are small relative to the target, not far past
    # it either.
    assert target <= len(text) <= target * 1.05


def test_generate_document_text_is_deterministic_for_the_same_seed_and_index():
    first = _generate_document_text(7, seed=42, pages=5)
    second = _generate_document_text(7, seed=42, pages=5)

    assert first == second


def test_generate_document_text_differs_across_documents_and_seeds():
    doc_a = _generate_document_text(0, seed=0, pages=5)
    doc_b = _generate_document_text(1, seed=0, pages=5)
    doc_c = _generate_document_text(0, seed=1, pages=5)

    assert doc_a != doc_b
    assert doc_a != doc_c


def test_generate_document_text_has_no_duplicate_chunk_sized_windows():
    # This is the exact pitfall README.md's calibration run already hit
    # once: near-identical chunk text let EmbeddingCache silently skip
    # re-embedding most chunks, inflating measured throughput ~2.7x. A
    # sliding 2000-char window (matching config.py's default
    # chunk_size_chars) must never repeat within one generated document.
    text = _generate_document_text(3, seed=0, pages=10)
    chunk_size = 2000

    windows = {
        text[start : start + chunk_size]
        for start in range(0, len(text) - chunk_size, chunk_size)
    }
    window_count = len(range(0, len(text) - chunk_size, chunk_size))

    assert len(windows) == window_count


def test_generate_document_text_has_no_duplicate_windows_across_documents():
    doc_a = _generate_document_text(0, seed=0, pages=3)
    doc_b = _generate_document_text(1, seed=0, pages=3)
    chunk_size = 2000

    windows_a = {doc_a[i : i + chunk_size] for i in range(0, len(doc_a) - chunk_size, chunk_size)}
    windows_b = {doc_b[i : i + chunk_size] for i in range(0, len(doc_b) - chunk_size, chunk_size)}

    assert windows_a.isdisjoint(windows_b)


def test_tier_for_index_splits_documents_evenly_across_tiers():
    document_count = 9

    assignments = [
        _tier_for_index(i, document_count, ACCESS_TIERS) for i in range(document_count)
    ]

    assert assignments.count(TIER_EMPLOYEE) == 3
    assert assignments.count(TIER_MANAGER) == 3
    assert assignments.count(TIER_DIRECTOR) == 3


def test_tier_for_index_handles_a_count_not_evenly_divisible_by_tier_count():
    document_count = 10

    assignments = [
        _tier_for_index(i, document_count, ACCESS_TIERS) for i in range(document_count)
    ]

    # Every index must land in a real tier - no assignment falls off the
    # end when document_count isn't a multiple of len(tiers).
    assert all(tier in ACCESS_TIERS for tier in assignments)
    assert len(assignments) == document_count


def test_generate_corpus_writes_the_requested_number_of_documents(tmp_path: Path):
    generate_corpus(
        tmp_path,
        document_count=6,
        pages_per_document=1,
        access_tiers=ACCESS_TIERS,
        seed=0,
    )

    written = list(tmp_path.glob("*/*.md"))
    assert len(written) == 6


def test_generate_corpus_distributes_documents_across_tier_subfolders(tmp_path: Path):
    generate_corpus(
        tmp_path,
        document_count=6,
        pages_per_document=1,
        access_tiers=ACCESS_TIERS,
        seed=0,
    )

    assert len(list((tmp_path / TIER_EMPLOYEE).glob("*.md"))) == 2
    assert len(list((tmp_path / TIER_MANAGER).glob("*.md"))) == 2
    assert len(list((tmp_path / TIER_DIRECTOR).glob("*.md"))) == 2


def test_generate_corpus_is_reproducible_from_the_same_seed(tmp_path: Path):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    generate_corpus(first_output, document_count=3, pages_per_document=1, seed=5)
    generate_corpus(second_output, document_count=3, pages_per_document=1, seed=5)

    first_files = sorted(first_output.glob("*/*.md"))
    second_files = sorted(second_output.glob("*/*.md"))
    assert [f.read_text() for f in first_files] == [f.read_text() for f in second_files]
