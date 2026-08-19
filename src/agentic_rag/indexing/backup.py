from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


# Qdrant's local/embedded mode holds an exclusive OS-level advisory lock
# (via portalocker) on this file for the entire lifetime of the QdrantClient
# that opened storage_path - which, in this codebase, is the whole life of
# the app process (api/app.py's lifespan holds one client open throughout).
# shutil.copytree() tries to open every file it copies; on Windows this
# collides with that exclusive lock and raises PermissionError - reproduced
# directly against this project's own live, still-open client, not a
# theoretical concern. Excluding it from the copy is safe: the lock file
# carries no data of its own (it's a pure OS-lock marker), and Qdrant
# creates a fresh one automatically the next time any QdrantClient opens
# the backup path - restoring from a backup with no `.lock` file works
# identically to one that has it, live-verified the same way.
_IGNORE_LOCK_FILE = shutil.ignore_patterns(".lock")


def backup_qdrant_storage(storage_path: Path, backup_dir: Path, *, retention_count: int) -> Path:
    """Copy `storage_path` (embedded Qdrant's on-disk collection data) into
    a new timestamped subdirectory under `backup_dir`, then prune anything
    beyond the `retention_count` most recent backups.

    Qdrant's own snapshot API (`client.create_snapshot()`) deliberately
    isn't used here - it raises `NotImplementedError` for local/embedded
    mode ("Snapshots are not supported in the local Qdrant. Please use
    server Qdrant if you need full snapshots.", confirmed directly against
    `qdrant_client.local.qdrant_local.QdrantLocal`), and this project runs
    embedded mode specifically because Docker isn't available in this dev
    environment (`qdrant_setup.get_client()`'s own docstring). A plain
    filesystem copy of the storage directory is the only backup mechanism
    that actually works for this deployment mode.

    This exists to bound *whole-index* loss - a corrupted on-disk store, a
    bad shutdown, a disk issue - not to protect against a single
    accidentally-deleted document (that case is already cheap to recover
    from: the watched folder, not Qdrant, is this system's source of
    truth, so the next sync cycle just re-ingests the one file - see
    `ingestion/scheduler.py`). A whole-index rebuild is the case this
    genuinely doesn't want to force, because this project's own real
    10,000-document load test (`loadtest/README.md`) never finished a
    from-scratch rebuild at that scale on this hardware - "just re-index
    everything" is not a cheap fallback once the corpus is large.

    Copied atomically: to a uniquely-suffixed `.tmp` directory first, then
    renamed into its final name (the *same* unique suffix, not just the
    timestamp) - `Path.replace()` is an atomic filesystem rename for
    directories on both POSIX and Windows when the destination doesn't
    already exist. The unique suffix is on *both* names specifically so two
    calls that happen to generate an identical microsecond timestamp still
    get genuinely distinct final directories instead of the second call's
    `replace()` colliding with the first's already-existing final path - an
    earlier version of this function only suffixed the tmp name, which
    protected the copy step but not the actual collision point.

    If the copy itself fails partway (disk full, the permission error
    excluding `.lock` doesn't already cover, an interrupted process), the
    partial `tmp_path` is removed before the exception propagates - a
    failed backup must not leave a near-full-size, silently-orphaned copy
    of the index on disk that nothing ever cleans up (an earlier version
    of this function had exactly that gap: `.tmp` directories are excluded
    from `list_backups()` by design, since an incomplete backup must never
    be counted as restorable, but that also meant an incomplete one was
    invisible to `_prune_old_backups()` and simply leaked, unbounded, on
    every subsequent failed attempt).

    Restoring from a backup is a manual operator action, not something
    this function (or any code) does automatically: the running process
    holds the live storage path open, so safely replacing it requires
    stopping the process first - see `docs/REQUIREMENTS.md`'s Miscellaneous
    Discussions section for the documented restore procedure.
    """
    if not storage_path.exists():
        raise FileNotFoundError(f"Qdrant storage path does not exist: {storage_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    unique = f"{timestamp}Z-{uuid.uuid4().hex[:8]}"
    final_path = backup_dir / unique
    tmp_path = backup_dir / f"{unique}.tmp"

    try:
        shutil.copytree(storage_path, tmp_path, ignore=_IGNORE_LOCK_FILE)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    tmp_path.replace(final_path)

    _prune_old_backups(backup_dir, retention_count)
    return final_path


def list_backups(backup_dir: Path) -> list[Path]:
    """Every completed backup under `backup_dir`, oldest first (sorted by
    directory name, which is a timestamp - see `backup_qdrant_storage()`).
    Empty list if `backup_dir` doesn't exist yet (no backup has ever run).

    Directories ending in `.tmp` - a backup interrupted mid-copy - are
    excluded: an incomplete backup must never be counted as a real,
    restorable one.
    """
    if not backup_dir.exists():
        return []
    return sorted(
        (p for p in backup_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")),
        key=lambda p: p.name,
    )


def _prune_old_backups(backup_dir: Path, retention_count: int) -> None:
    # `backups[:-retention_count]` is a footgun at retention_count == 0:
    # Python's `-0 == 0`, so `backups[:-0]` means `backups[:0]` - an empty
    # slice, not "the whole list" - meaning a retention_count of 0 would
    # silently prune *nothing* instead of everything. Computing the keep
    # count explicitly avoids relying on negative-slice arithmetic staying
    # correct at that boundary. (Settings itself rejects retention_count
    # <= 0 via `Field(gt=0)`, but this function takes a bare int with no
    # such guarantee, so it should be correct on its own terms too.)
    keep_count = max(retention_count, 0)
    backups = list_backups(backup_dir)
    stale = backups[: len(backups) - keep_count] if len(backups) > keep_count else []
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
