import pytest

from agentic_rag.ingestion.tagger import (
    UnknownAccessTierError,
    UntaggedDocumentError,
    access_tier_for,
)
from tests.access_tiers import ACCESS_TIERS, TIER_DIRECTOR, TIER_EMPLOYEE, TIER_MANAGER



def test_access_tier_for_returns_first_path_segment():
    assert access_tier_for("manager/report.txt", ACCESS_TIERS) == TIER_MANAGER


def test_access_tier_for_supports_nested_paths_within_a_tier():
    assert access_tier_for("employee/subfolder/report.txt", ACCESS_TIERS) == TIER_EMPLOYEE


def test_access_tier_for_normalizes_windows_style_backslash_paths():
    assert access_tier_for("director\\sub\\report.txt", ACCESS_TIERS) == TIER_DIRECTOR


def test_access_tier_for_raises_when_file_has_no_tier_folder():
    with pytest.raises(UntaggedDocumentError):
        access_tier_for("report.txt", ACCESS_TIERS)


def test_access_tier_for_raises_for_unknown_tier():
    with pytest.raises(UnknownAccessTierError):
        access_tier_for("not-a-tier/report.txt", ACCESS_TIERS)
