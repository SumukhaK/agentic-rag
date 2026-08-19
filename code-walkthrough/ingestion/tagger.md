# `ingestion/tagger.py`

**Purpose:** This file is responsible for figuring out which "access tier" (permission level — e.g. "manager" versus "employee") a document belongs to, purely from where it sits in the watched folder's directory structure. The system's convention is that every document must live inside a top-level subfolder named after its tier, so the folder structure itself is the source of truth for document permissions — no separate metadata file or database entry is needed. This keeps access control simple and auditable: to see what tier a document is in, you just look at its path. This file also defines the specific errors raised when a document doesn't follow that convention (sitting directly at the root with no tier folder, or sitting in a folder name that isn't a recognized tier).

## Line-by-line walkthrough

### Line 1 — Import
```python
from pathlib import PurePosixPath
```
- Imports `PurePosixPath`, a path-manipulation class that always uses forward-slash (`/`) semantics regardless of the operating system. It's used (rather than plain `Path`) because this function works on a path string that may have come from anywhere (including Windows-style backslash paths, normalized below) and only needs pure string/segment manipulation — it never touches the actual filesystem, so there's no need for the OS-specific behavior of `Path`.

### Lines 4-10 — `UntaggedDocumentError`
```python
class UntaggedDocumentError(Exception):
    """Raised when a document sits directly under the watched root, with no
    tier subfolder to derive its access tier from."""

    def __init__(self, relative_path: str):
        super().__init__(f"'{relative_path}' has no tier subfolder")
        self.relative_path = relative_path
```
- `class UntaggedDocumentError(Exception):` — a custom exception type (rather than reusing a generic `ValueError`) so callers further up the chain (like `ingestion/pipeline.py`'s broad `except Exception`) can distinguish this specific failure kind if they ever need to, and so its error message is purpose-built and clear.
- The docstring states exactly when this is raised: a document with no tier subfolder above it at all (i.e. it sits directly at the root of the watched folder).
- `def __init__(self, relative_path: str):` — the constructor takes the offending file's relative path.
- `super().__init__(f"'{relative_path}' has no tier subfolder")` — calls the base `Exception.__init__` with a formatted, human-readable message, so printing/logging the exception (e.g. via `str(exc)`) gives a clear description.
- `self.relative_path = relative_path` — also stores the path as an attribute on the exception object itself, so calling code can programmatically inspect which file failed (not just read the message string).

### Lines 13-19 — `UnknownAccessTierError`
```python
class UnknownAccessTierError(Exception):
    """Raised when a document's tier folder isn't in the configured tier list."""

    def __init__(self, tier: str, relative_path: str):
        super().__init__(f"unknown access tier '{tier}' for '{relative_path}'")
        self.tier = tier
        self.relative_path = relative_path
```
- `class UnknownAccessTierError(Exception):` — a second custom exception type, for a different failure mode: the document *does* sit under a subfolder, but that subfolder's name isn't one of the tiers the system knows about (e.g. a typo'd folder name, or a folder that was never configured as a valid tier).
- `def __init__(self, tier: str, relative_path: str):` — takes both the unrecognized tier name and the file's path.
- `super().__init__(...)` — builds a clear message naming both the bad tier and the file it was found on.
- `self.tier = tier` / `self.relative_path = relative_path` — stores both pieces of information as attributes for programmatic access.

### Lines 22-36 — `access_tier_for` function
```python
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
```
- `def access_tier_for(relative_path: str, known_tiers: list[str]) -> str:` — the public function: given a file's relative path and the list of tier names the system is configured to accept, returns the tier string, or raises one of the two errors above.
- The docstring restates the folder-naming convention with concrete examples (`manager/report.txt`, `employee/subfolder/report.txt`) to make the rule unambiguous.
- `relative_path.replace("\\", "/")` — normalizes Windows-style backslash path separators to forward slashes first, so this function behaves the same regardless of which OS produced the path string (important since the codebase runs on both Windows, per this environment, and POSIX systems).
- `PurePosixPath(...).parts` — splits the normalized path into its individual segments as a tuple, e.g. `"manager/subfolder/report.txt"` becomes `("manager", "subfolder", "report.txt")`.
- `if len(parts) < 2:` — if there are fewer than 2 segments, the file sits directly at the watched folder's root with no enclosing tier folder at all (a single segment is just the filename itself).
- `raise UntaggedDocumentError(relative_path)` — in that case, raises the untagged-document error described above.
- `tier = parts[0]` — otherwise, the first path segment (the top-level folder name) is taken as the candidate tier.
- `if tier not in known_tiers:` — checks that candidate tier name against the caller-supplied list of valid, configured tiers.
- `raise UnknownAccessTierError(tier, relative_path)` — if it's not recognized, raises the second error type, naming both the bad tier and the file.
- `return tier` — if it passed both checks, the tier name is returned as this document's access tier.
