from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


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

    Copied atomically: to a `.tmp`-suffixed directory first, then renamed
    into its final timestamped name - `Path.replace()` is an atomic
    filesystem rename for directories on both POSIX and Windows when the
    destination doesn't already exist (it never does here, since every
    call gets a fresh timestamp), so a crash mid-copy leaves only an
    orphaned `.tmp` directory (ignored by `list_backups()`, overwritten by
    the next call using the same stale name only in the astronomically
    unlikely case of a same-microsecond timestamp collision - a random
    suffix is added specifically to avoid relying on that) rather than a
    half-written backup that looks complete.

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
    final_path = backup_dir / f"{timestamp}Z"
    tmp_path = backup_dir / f"{timestamp}Z-{uuid.uuid4().hex[:8]}.tmp"

    shutil.copytree(storage_path, tmp_path)
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
    backups = list_backups(backup_dir)
    stale = backups[:-retention_count] if len(backups) > retention_count else []
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
