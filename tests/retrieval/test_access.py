import pytest

from agentic_rag.retrieval.access import UnknownAccessTierError, allowed_tiers_for
from access_tiers import ACCESS_TIERS, TIER_DIRECTOR, TIER_EMPLOYEE, TIER_MANAGER

KNOWN_TIERS = ACCESS_TIERS


def test_allowed_tiers_for_includes_own_tier_and_everything_below():
    assert allowed_tiers_for(TIER_MANAGER, KNOWN_TIERS) == [TIER_EMPLOYEE, TIER_MANAGER]


def test_allowed_tiers_for_lowest_tier_includes_only_itself():
    assert allowed_tiers_for(TIER_EMPLOYEE, KNOWN_TIERS) == [TIER_EMPLOYEE]


def test_allowed_tiers_for_highest_tier_includes_everything():
    assert allowed_tiers_for(TIER_DIRECTOR, KNOWN_TIERS) == ACCESS_TIERS


def test_allowed_tiers_for_raises_for_an_unknown_user_tier():
    with pytest.raises(UnknownAccessTierError):
        allowed_tiers_for("not-a-tier", KNOWN_TIERS)
