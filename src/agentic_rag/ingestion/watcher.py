from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileState:
    """Cheap, deterministic fingerprint of a file's on-disk state."""

    size: int
    mtime_ns: int


@dataclass(frozen=True)
class FolderChanges:
    created: list[str]
    modified: list[str]
    deleted: list[str]


def snapshot(folder: Path) -> dict[str, FileState]:
    """Fingerprint every file under `folder`, keyed by path relative to it."""
    result: dict[str, FileState] = {}
    for path in Path(folder).rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(folder))
            stat = path.stat()
            result[relative] = FileState(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


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
