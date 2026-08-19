from pathlib import PurePosixPath


class UntaggedDocumentError(Exception):
    """Raised when a document sits directly under the watched root, with no
    tier subfolder to derive its access tier from."""

    def __init__(self, relative_path: str):
        super().__init__(f"'{relative_path}' has no tier subfolder")
        self.relative_path = relative_path


class UnknownAccessTierError(Exception):
    """Raised when a document's tier folder isn't in the configured tier list."""

    def __init__(self, tier: str, relative_path: str):
        super().__init__(f"unknown access tier '{tier}' for '{relative_path}'")
        self.tier = tier
        self.relative_path = relative_path


def access_tier_for(relative_path: str, known_tiers: list[str]) -> str:
    """Derive a document's access tier from its first path segment.

    Convention: every document lives under a top-level subfolder named after
    its tier, e.g. `manager/report.txt` or `employee/subfolder/report.txt`.
    """
    parts = PurePosixPath(relative_path.replace("\\", "/")).parts
    if len(parts) < 2:
        raise UntaggedDocumentError(relative_path)

    tier = parts[0]
    if tier not in known_tiers:
        raise UnknownAccessTierError(tier, relative_path)

    return tier
