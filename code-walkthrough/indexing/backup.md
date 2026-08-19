# `indexing/backup.py`

**Purpose:** This file protects against losing the *entire* Qdrant search index, not a single document. Since this project runs Qdrant in embedded (local, no-server) mode, Qdrant's own built-in backup feature ("snapshots") isn't available - so this file does the next best thing: it periodically copies the whole folder where Qdrant stores its data to a separate, timestamped backup folder, and automatically deletes old backups so they don't pile up forever. If the live index ever gets corrupted (a bad shutdown, a disk problem), a human operator can point the app at one of these backup folders instead of losing everything and having to re-process every document from scratch - which, at this project's real target scale, has already been shown (in a real load test) to be slow and unreliable.

## Line-by-line walkthrough

### Lines 1-6 — Imports
```python
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
```
- `from __future__ import annotations` — a Python feature that lets type hints (like `Path` in a function signature) be written without needing every referenced type imported at the very top in a strict order; mostly a convenience/consistency choice matching other files in this codebase.
- `import shutil` — Python's standard library for higher-level file operations; used here for `shutil.copytree` (copy an entire folder) and `shutil.rmtree` (delete an entire folder).
- `import uuid` — used to generate a short random string, so two backups that happen to start at the exact same moment (down to the microsecond) still get different temporary folder names.
- `from datetime import datetime, timezone` — used to build a timestamp for naming each backup folder.
- `from pathlib import Path` — the modern, object-oriented way Python represents file/folder paths, used throughout this function's signature and body.

### Lines 9-48 — `backup_qdrant_storage`: doing the actual backup
```python
def backup_qdrant_storage(storage_path: Path, backup_dir: Path, *, retention_count: int) -> Path:
    """..."""
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
```
- `def backup_qdrant_storage(storage_path: Path, backup_dir: Path, *, retention_count: int) -> Path:` — the function takes the folder to back up (`storage_path`, where Qdrant's live data lives), the folder to store backups in (`backup_dir`), and how many backups to keep (`retention_count`, required to be passed by name because of the `*`). It returns the path of the backup it just created.
- The long docstring (not fully reproduced here, but present in the real file) explains *why* this exists rather than using Qdrant's own snapshot feature: that feature was checked directly against the installed library and confirmed to raise an error in embedded/local mode, so a plain folder copy is the only option that actually works for how this project runs Qdrant. It also explains this is meant to guard against losing the *whole* index, not a single accidentally-deleted document (the watched folder already makes recovering a single document cheap - see `ingestion/scheduler.py`), and that restoring from a backup is something a human does manually, not something this code does automatically.
- `if not storage_path.exists(): raise FileNotFoundError(...)` — if there's nothing to back up (the Qdrant storage folder doesn't exist at all, perhaps because the app has never run yet), fail loudly and clearly rather than silently creating an empty, useless "backup."
- `backup_dir.mkdir(parents=True, exist_ok=True)` — makes sure the backup folder (and any parent folders it needs) exists, without complaining if it's already there.
- `timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")` — builds a sortable text timestamp (year, month, day, hour, minute, second, microsecond) using UTC time, so backups from different timezones are still comparable and so alphabetically sorting the backup folder names also sorts them by when they were made.
- `final_path = backup_dir / f"{timestamp}Z"` — the name this backup will have once it's complete (the trailing `Z` is a common convention meaning "this timestamp is in UTC").
- `tmp_path = backup_dir / f"{timestamp}Z-{uuid.uuid4().hex[:8]}.tmp"` — a *different*, temporary name (with a random 8-character suffix and a `.tmp` ending) used while the copy is still in progress. Using a different name for the in-progress copy, rather than writing directly to `final_path`, is what makes the next two lines safe.
- `shutil.copytree(storage_path, tmp_path)` — does the actual work: recursively copies every file and subfolder from the live Qdrant storage into the temporary backup location. This is the slow part, proportional to how much data is in the index.
- `tmp_path.replace(final_path)` — renames the temporary folder to its real, final name. A rename (as opposed to a copy) is treated by the operating system as a single, instant, all-or-nothing operation - so if the process crashes *during* the slow copy above, only the half-finished `.tmp` folder is left behind, and `final_path` never exists in a broken state. Readers only ever see a backup that's either fully there or not there at all.
- `_prune_old_backups(backup_dir, retention_count)` — after adding the new backup, deletes old ones beyond the configured limit (see below).
- `return final_path` — hands back the location of the backup that was just created, so a caller can log it.

### Lines 51-62 — `list_backups`: finding what backups already exist
```python
def list_backups(backup_dir: Path) -> list[Path]:
    """..."""
    if not backup_dir.exists():
        return []
    return sorted(
        (p for p in backup_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")),
        key=lambda p: p.name,
    )
```
- `def list_backups(backup_dir: Path) -> list[Path]:` — returns every completed backup, oldest first.
- `if not backup_dir.exists(): return []` — if backups have never been taken, there's nothing to list; return an empty list rather than raising an error.
- `sorted((p for p in backup_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")), key=lambda p: p.name)` — looks at every item directly inside `backup_dir`, keeps only the ones that are folders (not stray files) and whose name doesn't end in `.tmp` (excluding any leftover, never-finished backup from a crash mid-copy), then sorts them by name. Because the names are timestamps built the same way every time, sorting alphabetically is the same as sorting chronologically - oldest backup first.

### Lines 65-69 — `_prune_old_backups`: deleting backups beyond the retention limit
```python
def _prune_old_backups(backup_dir: Path, retention_count: int) -> None:
    backups = list_backups(backup_dir)
    stale = backups[:-retention_count] if len(backups) > retention_count else []
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
```
- `def _prune_old_backups(backup_dir: Path, retention_count: int) -> None:` — an internal helper (the leading underscore signals it's not meant to be called from outside this file) that deletes old backups once there are more than `retention_count` of them.
- `backups = list_backups(backup_dir)` — gets the current, oldest-first list of real backups.
- `stale = backups[:-retention_count] if len(backups) > retention_count else []` — if there are more backups than the limit allows, this takes everything *except* the last `retention_count` items (Python's negative slicing, `list[:-N]`, means "everything except the final N items") - i.e., the oldest excess ones. If there aren't more backups than the limit, nothing is considered stale.
- `for path in stale: shutil.rmtree(path, ignore_errors=True)` — deletes each stale backup folder and everything inside it. `ignore_errors=True` means a problem deleting one old backup (e.g. a file briefly locked by another process) doesn't stop the function or raise an exception - pruning is a housekeeping nicety, not something that should ever block or crash the backup that was just successfully created.
