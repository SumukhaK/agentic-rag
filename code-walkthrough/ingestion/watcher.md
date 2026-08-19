# `ingestion/watcher.py`

**Purpose:** This file answers the question "what changed in the watched folder since we last checked?" — which files are brand new, which were modified, and which were deleted. Rather than relying on the operating system's live file-change notifications (things like Windows' `ReadDirectoryChangesW` or Linux's `inotify`), this file takes a simpler, more robust approach: it takes a cheap "fingerprint" (size and modification time) of every file in the folder, and compares that fingerprint against the fingerprint taken at the last check. This "snapshot and diff" approach is deterministic and resilient — it doesn't depend on a background OS-level listener staying alive and correctly delivering every event (which can silently drop events, miss changes made while the listener wasn't running, or behave differently across operating systems); instead, the truth is always freshly recomputed by directly reading the actual state of the filesystem, so it inherently self-heals from things like process restarts or missed events.

## Line-by-line walkthrough

### Lines 1-4 — Imports
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
```
- `from __future__ import annotations` — defers evaluation of type hints so modern syntax like `dict[str, FileState]` works cleanly.
- `from dataclasses import dataclass` — imports the decorator used for the two simple data types below.
- `from pathlib import Path` — imports the path type used for walking the filesystem and representing the watched folder.

### Lines 7-12 — `FileState`
```python
@dataclass(frozen=True)
class FileState:
    """Cheap, deterministic fingerprint of a file's on-disk state."""

    size: int
    mtime_ns: int
```
- `@dataclass(frozen=True)` — an immutable data record, matching the pattern used throughout ingestion for "this represents a fact that was true at some point, don't let it be silently changed afterward."
- The docstring explains the design intent directly: this is meant to be a *cheap* check — reading a file's size and modification time from its filesystem metadata (`stat`) is nearly instant, unlike, say, hashing the file's full contents, which would be far more expensive to do for every file on every cycle.
- `size: int` — the file's size in bytes. Included because a file's modification time can sometimes be unreliable or coarse-grained on some filesystems, so pairing it with size gives a more robust (if still not perfect) fingerprint — a change that alters the file's size will always be caught even if the timestamp resolution or clock behaves oddly.
- `mtime_ns: int` — the file's last-modified time, in nanoseconds (`st_mtime_ns` from `os.stat`), chosen over the lower-precision second-based `st_mtime` to reduce the chance that two different states of a file could produce an identical fingerprint.

### Lines 15-19 — `FolderChanges`
```python
@dataclass(frozen=True)
class FolderChanges:
    created: list[str]
    modified: list[str]
    deleted: list[str]
```
- `@dataclass(frozen=True)` — another immutable record.
- `created: list[str]` — relative paths of files that exist now but didn't exist in the previous snapshot.
- `modified: list[str]` — relative paths of files that existed before and now, but whose fingerprint (size or mtime) changed.
- `deleted: list[str]` — relative paths of files that existed in the previous snapshot but are no longer present now.

### Lines 22-30 — `snapshot` function
```python
def snapshot(folder: Path) -> dict[str, FileState]:
    """Fingerprint every file under `folder`, keyed by path relative to it."""
    result: dict[str, FileState] = {}
    for path in Path(folder).rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(folder))
            stat = path.stat()
            result[relative] = FileState(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result
```
- `def snapshot(folder: Path) -> dict[str, FileState]:` — takes the watched folder's root path and returns a dictionary mapping every file's path (relative to that root) to its `FileState` fingerprint. This dictionary *is* the "snapshot" of the folder's current state.
- `result: dict[str, FileState] = {}` — the dictionary being built up.
- `for path in Path(folder).rglob("*"):` — recursively walks every entry (file or directory) under `folder`, at any depth — `rglob("*")` is `pathlib`'s recursive glob that matches everything.
- `if path.is_file():` — skips directories themselves (which `rglob` also yields), since only actual files need fingerprinting.
- `relative = str(path.relative_to(folder))` — converts the absolute path into a string path relative to the watched folder's root (e.g. `manager/report.txt`), which is the stable identifier used everywhere else in the ingestion pipeline (it doesn't change if the whole watched folder is moved to a different absolute location).
- `stat = path.stat()` — performs a single filesystem metadata read for the file (an `os.stat` call under the hood), giving access to its size and modification time.
- `result[relative] = FileState(size=stat.st_size, mtime_ns=stat.st_mtime_ns)` — stores this file's fingerprint in the result dictionary, keyed by its relative path.
- `return result` — returns the complete snapshot once every file has been walked.

### Lines 33-48 — `diff_snapshots` function
```python
def diff_snapshots(
    previous: dict[str, FileState], current: dict[str, FileState]
) -> FolderChanges:
    """Compare two snapshots of the same folder taken at different times."""
    previous_paths = set(previous)
    current_paths = set(current)

    created = sorted(current_paths - previous_paths)
    deleted = sorted(previous_paths - current_paths)
    modified = sorted(
        path
        for path in previous_paths & current_paths
        if previous[path] != current[path]
    )

    return FolderChanges(created=created, modified=modified, deleted=deleted)
```
- `def diff_snapshots(previous: dict[str, FileState], current: dict[str, FileState]) -> FolderChanges:` — takes two snapshots (typically: the snapshot saved from the last sync cycle, and a freshly-taken one from right now) and computes what changed between them.
- `previous_paths = set(previous)` / `current_paths = set(current)` — converts each snapshot's dictionary keys (the relative file paths) into sets, which makes the set-arithmetic operations below both fast and easy to read.
- `created = sorted(current_paths - previous_paths)` — set difference: paths present now but not before are newly created files. `sorted(...)` gives a deterministic, stable ordering (important for reproducibility — the same input snapshots should always produce the same output order, which also makes logs and tests deterministic rather than depending on arbitrary set-iteration order).
- `deleted = sorted(previous_paths - current_paths)` — the reverse set difference: paths that were present before but are gone now are deletions. This is the crucial mechanism by which a file's removal from disk is detected — there's no OS delete-event needed; if a path silently vanishes between two snapshots, this line catches it regardless of how or when it was removed.
- `modified = sorted(path for path in previous_paths & current_paths if previous[path] != current[path])` — set intersection (`&`) finds paths present in *both* snapshots (i.e., not new, not deleted), then the generator expression filters that down to only the ones whose `FileState` actually differs between the two snapshots (relying on the `FileState` dataclass's auto-generated `__eq__`, which compares `size` and `mtime_ns`) — meaning the file's content most likely changed since the last check.
- `return FolderChanges(created=created, modified=modified, deleted=deleted)` — packages the three categorized lists into the `FolderChanges` result, which is exactly what `ingestion/pipeline.py`'s `process_changes()` expects as its input describing what needs (re)processing.
