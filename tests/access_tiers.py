"""Single source of truth for the access-tier names used across the test
suite - a future rename only needs to change the three values here, not
every test file that references a tier.

This mirrors Settings.access_tiers' own default (config.py) and
DEFAULT_ACCESS_TIERS (loadtest/corpus_generator.py), which independently
hold the same three literal values. A test that must prove behavior is
driven by one of those specific sources - not just "the tier names",
e.g. tests/loadtest/test_runner.py's checks that the load test uses its
own fixed tier layout regardless of Settings.access_tiers - should import
that source directly rather than this module, so drift between the two
still fails the test.
"""

TIER_EMPLOYEE = "employee"
TIER_MANAGER = "manager"
TIER_DIRECTOR = "director"

ACCESS_TIERS = [TIER_EMPLOYEE, TIER_MANAGER, TIER_DIRECTOR]
