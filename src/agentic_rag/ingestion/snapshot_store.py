from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.ingestion.watcher import FileState


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
