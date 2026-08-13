class UnknownAccessTierError(Exception):
    """Raised when a user's tier isn't in the configured tier list."""


def allowed_tiers_for(user_tier: str, known_tiers: list[str]) -> list[str]:
    """Resolve which tiers `user_tier` may see, per the linear-tier access
    model (REQUIREMENTS.md §11): a user at a given tier sees content
    tagged at their tier or any tier below it.

    `known_tiers` is ordered lowest to highest (as configured in
    ACCESS_TIERS); the result preserves that order.
    """
    if user_tier not in known_tiers:
        raise UnknownAccessTierError(
            f"unknown user tier '{user_tier}'; known tiers: {known_tiers}"
        )
    return known_tiers[: known_tiers.index(user_tier) + 1]
