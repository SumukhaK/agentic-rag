# `ingestion/snapshot_store.py`

**Purpose:** This file persists the folder "snapshot" (the per-file size/modification-time fingerprints produced by `ingestion/watcher.py`) to disk between runs, and loads it back. This matters because the whole sync system compares "what the folder looks like now" against "what it looked like last time" to figure out what changed — and if that "last time" snapshot only lived in memory, restarting the process would forget it entirely. Without a persisted snapshot, a restart would either force a wasteful full re-index of every document, or worse, permanently fail to notice that a file was deleted while the process was down (since there'd be nothing in memory to compare against and notice the file missing). This file exists purely to make that memory durable, and to do so safely (without risking a corrupted file if the process crashes mid-write).

## Line-by-line walkthrough

### Lines 1-6 — Imports
```python
from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.ingestion.watcher import FileState
```
- `from __future__ import annotations` — defers type hint evaluation for modern syntax support.
- `import json` — used to serialize the snapshot dictionary to a JSON file on disk and parse it back.
- `from pathlib import Path` — the path type used for the snapshot file's location.
- `from agentic_rag.ingestion.watcher import FileState` — imports the `FileState` fingerprint type (defined in `watcher.py`) that this module reads and writes.

### Lines 9-30 — `load_snapshot` function
```python
def load_snapshot(path: Path) -> dict[str, FileState]:
    """Load a previously-persisted snapshot from `path`, or `{}` if it
    doesn't exist yet - a fresh corpus (or the first run of this feature
    against an existing one) has nothing persisted, matching
    `run_sync_loop()`'s own documented cold-start behavior of treating
    every file in the watched folder as new.

    Without this, a process restart would start `run_sync_loop()` from an
    empty in-memory snapshot every time - not just wasteful (a full
    corpus re-index), but a real correctness gap: `diff_snapshots({},
    current)` can never report anything as deleted (there's nothing in
    `previous` to be missing from `current`), so a file removed while the
    process was down would never be detected as deleted at all, not even
    on the very next cycle after restart.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        relative_path: FileState(size=entry["size"], mtime_ns=entry["mtime_ns"])
        for relative_path, entry in raw.items()
    }
```
- `def load_snapshot(path: Path) -> dict[str, FileState]:` — takes the file path where a snapshot is (or would be) stored, and returns the snapshot as a dictionary of relative paths to `FileState` fingerprints — the same shape `watcher.py`'s `snapshot()` function produces.
- The docstring explains two things: first, that a missing file simply means "nothing persisted yet," which is treated the same way `run_sync_loop()` treats a fresh start — every file currently in the folder gets treated as newly created and gets indexed once. Second, and more important, it explains *why* persistence is necessary at all, not just a nice-to-have: without it, every restart would effectively reset to "no memory of what existed before," and since a deletion can only be detected by noticing a path that *was* in the previous snapshot is now missing, an always-empty starting snapshot means deletions that happened while the process was offline would be permanently invisible — the diff logic has no record that the file ever existed to notice it's now gone.
- `if not path.exists():` — checks whether a snapshot file exists at all yet.
- `return {}` — if not, returns an empty dictionary, representing "no prior knowledge," consistent with the cold-start behavior described above.
- `raw = json.loads(path.read_text())` — otherwise, reads the file's full text content and parses it as JSON into a plain Python dictionary (`raw`), whose values are plain untyped dicts like `{"size": ..., "mtime_ns": ...}` rather than `FileState` objects (JSON has no concept of Python dataclasses).
- `return {relative_path: FileState(size=entry["size"], mtime_ns=entry["mtime_ns"]) for relative_path, entry in raw.items()}` — rebuilds a proper `dict[str, FileState]` from the raw JSON data, reconstructing each `FileState` object from its `size` and `mtime_ns` fields. This is the step that converts the file's generic JSON shape back into the typed structure the rest of the ingestion code expects to work with.

### Lines 33-51 — `save_snapshot` function
```python
def save_snapshot(path: Path, snapshot: dict[str, FileState]) -> None:
    """Persist `snapshot` to `path`, atomically.

    Writes to a temp file in the same directory, then renames it into
    place - `Path.replace()` is an atomic filesystem rename on both POSIX
    and Windows, so a crash mid-write leaves the previous, still-valid
    snapshot in place rather than a truncated/corrupt one that would be
    misread as empty (and, per `load_snapshot()`'s docstring, silently
    lose deletion detection) on the next startup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        relative_path: {"size": state.size, "mtime_ns": state.mtime_ns}
        for relative_path, state in snapshot.items()
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(path)
```
- `def save_snapshot(path: Path, snapshot: dict[str, FileState]) -> None:` — takes the destination file path and the current in-memory snapshot dictionary, and writes it to disk. Returns nothing; success is implied by not raising.
- The docstring explains the "atomic write" technique used and exactly why it matters: if the process crashed or was killed partway through writing the snapshot file directly, the file on disk could be left half-written — and per `load_snapshot()`'s own logic, a broken/unparseable file would likely be treated as empty or cause a crash on the next load, which (as explained above) would silently break deletion detection going forward. Writing to a separate temporary file first, then atomically renaming it over the real target, avoids that: at every point in time, the real snapshot file on disk is either the old, fully-valid version or the new, fully-valid version — never something in between.
- `path.parent.mkdir(parents=True, exist_ok=True)` — ensures the directory that should contain the snapshot file actually exists, creating any missing parent directories along the way (`parents=True`), and not raising an error if it already exists (`exist_ok=True`). This matters on a first-ever run where the directory holding the snapshot file might not exist yet.
- `payload = {relative_path: {"size": state.size, "mtime_ns": state.mtime_ns} for relative_path, state in snapshot.items()}` — converts the `FileState` dataclass instances back into plain dictionaries, since the built-in `json` module doesn't know how to serialize arbitrary Python objects like dataclasses directly — it only understands basic types (dicts, lists, strings, numbers).
- `tmp_path = path.with_suffix(path.suffix + ".tmp")` — computes a sibling temporary file path by appending `.tmp` to the target file's existing suffix (e.g. `snapshot.json` becomes `snapshot.json.tmp`), placing it in the exact same directory as the final destination. Using the same directory (rather than, say, a system temp folder) is important because the atomic rename trick below only works reliably when source and destination are on the same filesystem/volume.
- `tmp_path.write_text(json.dumps(payload))` — serializes the plain-dict `payload` to a JSON string and writes it out fully to the temporary file.
- `tmp_path.replace(path)` — atomically renames the temp file to the real target path, replacing whatever was there before in a single, uninterruptible filesystem operation. This is the actual "atomic" part of the operation — `Path.replace()` (backed by `os.replace()`) is guaranteed atomic on both POSIX and Windows, meaning any process (or observer) will only ever see either the complete old file or the complete new file, never a partial one.
