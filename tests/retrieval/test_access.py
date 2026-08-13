import pytest

from agentic_rag.retrieval.access import UnknownAccessTierError, allowed_tiers_for

KNOWN_TIERS = ["tier-1", "tier-2", "tier-3"]


def test_allowed_tiers_for_includes_own_tier_and_everything_below():
    assert allowed_tiers_for("tier-2", KNOWN_TIERS) == ["tier-1", "tier-2"]


def test_allowed_tiers_for_lowest_tier_includes_only_itself():
    assert allowed_tiers_for("tier-1", KNOWN_TIERS) == ["tier-1"]


def test_allowed_tiers_for_highest_tier_includes_everything():
    assert allowed_tiers_for("tier-3", KNOWN_TIERS) == ["tier-1", "tier-2", "tier-3"]


def test_allowed_tiers_for_raises_for_an_unknown_user_tier():
    with pytest.raises(UnknownAccessTierError):
        allowed_tiers_for("not-a-tier", KNOWN_TIERS)
