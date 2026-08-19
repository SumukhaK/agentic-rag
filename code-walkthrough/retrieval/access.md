# `retrieval/access.py`

**Purpose:** This file answers one narrow question: "given a user's access tier, which tiers of content are they allowed to see?" The system uses a simple linear ranking of access tiers (for example, something like `public < internal < confidential`), and a user at a given tier is allowed to see anything at their tier or below. This file is the single place that encodes that rule, so every part of the system that needs to filter search results by permission (like `search.py`) calls into this one function instead of re-implementing the tier logic themselves. Keeping it here means the access rule only has to be gotten right once, and it fails loudly (raises an exception) if it's ever asked about a tier it doesn't recognize, rather than quietly letting someone see content they shouldn't.

## Line-by-line walkthrough

### Lines 1-2 — Custom exception for unknown tiers
```python
class UnknownAccessTierError(Exception):
    """Raised when a user's tier isn't in the configured tier list."""
```
- `class UnknownAccessTierError(Exception):` — defines a dedicated exception type instead of raising a generic `Exception` or `ValueError`. This lets calling code (or logging/monitoring) specifically catch or identify "someone was assigned an access tier that doesn't exist in our configuration," which is a distinct, actionable failure mode from other kinds of errors.
- The docstring documents when this exception fires, so anyone catching it (or reading a stack trace) immediately understands the failure without digging into the source.

### Lines 5-17 — `allowed_tiers_for`: resolving which tiers a user can see
```python
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
```
- `def allowed_tiers_for(user_tier: str, known_tiers: list[str]) -> list[str]:` — declares the function's contract: given a single tier name (`user_tier`) and the full ordered list of valid tiers (`known_tiers`), it returns the list of tiers that user is permitted to see. Type annotations make the expected shapes explicit for anyone calling this.
- The docstring explains the underlying business rule (a linear hierarchy where higher tiers can see everything at or below their level) and points to the requirements document (`REQUIREMENTS.md §11`) as the source of truth for that rule, so a reader knows this isn't an arbitrary design choice. It also calls out an important precondition: `known_tiers` must already be sorted from lowest to highest access, because the function relies entirely on list position to determine "below."
- `if user_tier not in known_tiers:` — checks whether the given user tier actually exists in the configured list of tiers before doing anything else.
- `raise UnknownAccessTierError(...)` — if the tier isn't recognized, the function immediately raises the custom exception defined above, with a message that includes both the bad tier and the full list of valid tiers (useful for debugging misconfiguration). This is a deliberate "fail loudly" design: silently treating an unknown tier as "no access" or "full access" could either wrongly deny a legitimate user or, worse, wrongly expose restricted content, so the code refuses to guess.
- `return known_tiers[: known_tiers.index(user_tier) + 1]` — this is the core logic. `known_tiers.index(user_tier)` finds the position of the user's tier in the ordered list (e.g. index 1 if `known_tiers = ["public", "internal", "confidential"]` and `user_tier = "internal"`). Slicing `known_tiers[:index + 1]` takes everything from the start of the list up to and including that position — i.e., the user's own tier plus every tier "below" it in the list order. The `+ 1` is needed because Python slice upper bounds are exclusive, so without it the user's own tier would be cut off. Because the slice starts from index 0 of the original list, the result naturally preserves the same lowest-to-highest ordering as `known_tiers`, exactly as the docstring promises. This purely index-based approach is simple and fast (no per-tier comparison logic needed) precisely because the access model is a strict linear hierarchy rather than something more complex like overlapping roles.
