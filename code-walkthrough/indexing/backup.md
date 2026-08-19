# `indexing/backup.py`

**Purpose:** This file protects against losing the *entire* Qdrant search index, not a single document. Since this project runs Qdrant in embedded (local, no-server) mode, Qdrant's own built-in backup feature ("snapshots") isn't available - so this file does the next best thing: it periodically copies the whole folder where Qdrant stores its data to a separate, timestamped backup folder, and automatically deletes old backups so they don't pile up forever. If the live index ever gets corrupted (a bad shutdown, a disk problem), a human operator can point the app at one of these backup folders instead of losing everything and having to re-process every document from scratch - which, at this project's real target scale, has already been shown (in a real load test) to be slow and unreliable.

This file went through a real self-review pass that caught 4 genuine bugs in its first version before it shipped - most importantly, that the very first version of this function would have silently failed on *every single real backup attempt*, because it tried to copy a file that Qdrant keeps permanently locked while the app is running. That fix (and the others) are called out explicitly below, not glossed over, because the reasoning behind them is exactly the kind of thing worth understanding about this file.

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
- `import shutil` — Python's standard library for higher-level file operations; used here for `shutil.copytree` (copy an entire folder), `shutil.rmtree` (delete an entire folder), and `shutil.ignore_patterns` (see the next section).
- `import uuid` — used to generate a short random string, giving every backup's temporary and final directory names a genuinely unique suffix.
- `from datetime import datetime, timezone` — used to build a timestamp for naming each backup folder.
- `from pathlib import Path` — the modern, object-oriented way Python represents file/folder paths, used throughout this function's signature and body.

### Lines 9-21 — Why the `.lock` file is deliberately skipped
```python
_IGNORE_LOCK_FILE = shutil.ignore_patterns(".lock")
```
This one line is the fix for the most serious bug this file's first version had. Qdrant's embedded/local mode creates a file named `.lock` inside its storage folder and holds an exclusive, operating-system-level lock on it for as long as the app is running (the same `QdrantClient` object is created once, at app startup, and kept open the whole time). `shutil.copytree()` normally tries to open and copy *every* file in the folder it's copying, including that one - and on Windows, trying to open a file someone else already has exclusively locked raises a `PermissionError`. This was reproduced directly: creating a real Qdrant collection, writing data to it, and then calling the backup function *while that same client was still open* (exactly what happens in production, since `ingestion/scheduler.py`'s background loop runs backups while the app's one shared client is alive) failed every time, with the app quietly logging the failure and never actually completing a backup. `shutil.ignore_patterns(".lock")` builds a filter that tells `copytree()` "skip any file named `.lock`" - which is safe to do, because the lock file itself holds no real data; it exists purely to enforce the operating-system lock, and Qdrant creates a fresh one automatically the moment any `QdrantClient` (including a restore) opens a folder that doesn't already have one.

### Lines 24-95 — `backup_qdrant_storage`: doing the actual backup
```python
def backup_qdrant_storage(storage_path: Path, backup_dir: Path, *, retention_count: int) -> Path:
    """..."""
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
```
- `def backup_qdrant_storage(storage_path: Path, backup_dir: Path, *, retention_count: int) -> Path:` — the function takes the folder to back up (`storage_path`, where Qdrant's live data lives), the folder to store backups in (`backup_dir`), and how many backups to keep (`retention_count`, required to be passed by name because of the `*`). It returns the path of the backup it just created.
- The long docstring (not fully reproduced here, but present in the real file) explains *why* this exists rather than using Qdrant's own snapshot feature - that feature was checked directly against the installed library and confirmed to raise an error in embedded/local mode - and explains this is meant to guard against losing the *whole* index, not a single accidentally-deleted document (the watched folder already makes recovering a single document cheap - see `ingestion/scheduler.py`). It also documents, in detail, the three bugs this function's first version had and how each was fixed - the same three explained inline below.
- `if not storage_path.exists(): raise FileNotFoundError(...)` — if there's nothing to back up (the Qdrant storage folder doesn't exist at all, perhaps because the app has never run yet), fail loudly and clearly rather than silently creating an empty, useless "backup."
- `backup_dir.mkdir(parents=True, exist_ok=True)` — makes sure the backup folder (and any parent folders it needs) exists, without complaining if it's already there.
- `timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")` — builds a sortable text timestamp (year, month, day, hour, minute, second, microsecond) using UTC time, so backups from different timezones are still comparable and so alphabetically sorting the backup folder names also sorts them by when they were made.
- `unique = f"{timestamp}Z-{uuid.uuid4().hex[:8]}"` — combines the timestamp with a short random suffix into one shared "unique" identifier. This is the fix for the second bug the first version had: that version put the random suffix only on the *temporary* folder's name, not the *final* one, meaning two backups that happened to be created in the exact same microsecond would still both try to claim the same final directory name — the first would succeed, and the second would fail when trying to rename into a name that already existed. Sharing one `unique` string between both names closes that gap completely: it's now genuinely impossible for two calls to collide on either name.
- `final_path = backup_dir / unique` and `tmp_path = backup_dir / f"{unique}.tmp"` — the final, "this backup is complete" name and the temporary, "still being written" name, both built from the same unique identifier.
- `try: shutil.copytree(storage_path, tmp_path, ignore=_IGNORE_LOCK_FILE)` — does the actual work: recursively copies every file and subfolder from the live Qdrant storage into the temporary backup location, except `.lock` (see above). This is the slow part, proportional to how much data is in the index.
- `except Exception: shutil.rmtree(tmp_path, ignore_errors=True); raise` — this is the fix for the third bug: if the copy fails partway through for any reason (the disk fills up, a permissions problem, the process gets killed), the partially-copied `tmp_path` folder is deleted immediately before the error is allowed to propagate to the caller. The first version of this function didn't do this - a failed copy left behind a large, incomplete folder that nothing else in the codebase ever looked at or cleaned up (since incomplete backups are deliberately excluded from the list of "real" backups - see `list_backups()` below), so it would just sit there forever, wasting disk space, and a backup that failed repeatedly (e.g. because the disk genuinely was full) would make the problem worse with every attempt. `raise` (with no arguments) re-raises the exact same exception that was caught, preserving its original type and message for whoever called this function.
- `tmp_path.replace(final_path)` — renames the temporary folder to its real, final name, but only once the copy has fully succeeded. A rename (as opposed to a copy) is treated by the operating system as a single, instant, all-or-nothing operation - so if the process crashes at this exact instant, only one of the two names ever really "exists" at once from an outside observer's point of view.
- `_prune_old_backups(backup_dir, retention_count)` — after adding the new backup, deletes old ones beyond the configured limit (see below).
- `return final_path` — hands back the location of the backup that was just created, so a caller can log it.

### Lines 98-112 — `list_backups`: finding what backups already exist
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
- `sorted((p for p in backup_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")), key=lambda p: p.name)` — looks at every item directly inside `backup_dir`, keeps only the ones that are folders (not stray files) and whose name doesn't end in `.tmp` (excluding any leftover, never-finished backup from a crash), then sorts them by name. Because the names all start with the same timestamp format, sorting alphabetically is the same as sorting chronologically - oldest backup first.

### Lines 115-129 — `_prune_old_backups`: deleting backups beyond the retention limit
```python
def _prune_old_backups(backup_dir: Path, retention_count: int) -> None:
    keep_count = max(retention_count, 0)
    backups = list_backups(backup_dir)
    stale = backups[: len(backups) - keep_count] if len(backups) > keep_count else []
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
```
- `def _prune_old_backups(backup_dir: Path, retention_count: int) -> None:` — an internal helper (the leading underscore signals it's not meant to be called from outside this file) that deletes old backups once there are more than `retention_count` of them.
- `keep_count = max(retention_count, 0)` — this line is the fix for the fourth bug. The first version of this function computed which backups to delete using `backups[:-retention_count]` - Python list slicing with a *negative* number counting from the end. That works fine for any `retention_count` of 1 or more, but breaks in a subtle way at exactly 0: in Python, `-0` is just `0`, so `backups[:-0]` doesn't mean "everything except the last zero items" (i.e., everything) - it actually means `backups[:0]`, an *empty* slice, meaning nothing would ever be identified as stale. A `retention_count` of 0 was meant to mean "keep no backups at all," but the old code would have silently kept every single one forever instead. Computing `keep_count` explicitly (clamped to never go below zero) and using it in ordinary arithmetic, rather than relying on negative-slice syntax, sidesteps that boundary case entirely. (In practice, the application's settings never allow `retention_count` to reach 0 - `config.py` requires it to be greater than zero - so this bug couldn't have been triggered through normal use of the app; it's fixed anyway because this function doesn't itself guarantee that precondition, and a future caller that skipped `Settings`' validation could have hit it.)
- `backups = list_backups(backup_dir)` — gets the current, oldest-first list of real backups.
- `stale = backups[: len(backups) - keep_count] if len(backups) > keep_count else []` — if there are more backups than the limit allows, this takes everything from the start of the list up to (but not including) the last `keep_count` items - i.e., the oldest excess ones. If there aren't more backups than the limit, nothing is considered stale.
- `for path in stale: shutil.rmtree(path, ignore_errors=True)` — deletes each stale backup folder and everything inside it. `ignore_errors=True` means a problem deleting one old backup (e.g. a file briefly locked by another process) doesn't stop the function or raise an exception - pruning is a housekeeping nicety, not something that should ever block or crash the backup that was just successfully created.
