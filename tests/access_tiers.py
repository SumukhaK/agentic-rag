"""Single source of truth for the access-tier names used across the test
suite, mirroring Settings.access_tiers' own default (config.py) - a future
rename only needs to change the three values here, not every test file
that references a tier.
"""

TIER_EMPLOYEE = "employee"
TIER_MANAGER = "manager"
TIER_DIRECTOR = "director"

ACCESS_TIERS = [TIER_EMPLOYEE, TIER_MANAGER, TIER_DIRECTOR]
