import pytest

from agentic_rag.ingestion.tagger import (
    UnknownAccessTierError,
    UntaggedDocumentError,
    access_tier_for,
)

KNOWN_TIERS = ["tier-1", "tier-2", "tier-3"]


def test_access_tier_for_returns_first_path_segment():
    assert access_tier_for("tier-2/report.txt", KNOWN_TIERS) == "tier-2"


def test_access_tier_for_supports_nested_paths_within_a_tier():
    assert access_tier_for("tier-1/subfolder/report.txt", KNOWN_TIERS) == "tier-1"


def test_access_tier_for_normalizes_windows_style_backslash_paths():
    assert access_tier_for("tier-3\\sub\\report.txt", KNOWN_TIERS) == "tier-3"


def test_access_tier_for_raises_when_file_has_no_tier_folder():
    with pytest.raises(UntaggedDocumentError):
        access_tier_for("report.txt", KNOWN_TIERS)


def test_access_tier_for_raises_for_unknown_tier():
    with pytest.raises(UnknownAccessTierError):
        access_tier_for("not-a-tier/report.txt", KNOWN_TIERS)
