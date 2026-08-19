# `ingestion/sync.py`

**Purpose:** This file ties together the watcher (which detects what changed on disk) and the pipeline (which converts, chunks, tags, and validates those changes) into a single "run one ingestion cycle" operation. It answers the question "what changed since last time, and what does that mean once fully processed?" for the *ingestion side only* — it deliberately does not touch the search index itself (that's `ingestion/scheduler.py`'s job, specifically `run_sync_cycle()`). Keeping this file scoped to just detection + processing (and not indexing) keeps its responsibility narrow and makes it independently testable without needing a real Qdrant connection.

## Line-by-line walkthrough

### Lines 1-11 — Imports
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.pipeline import (
    IngestedDocument,
    IngestionFailure,
    process_changes,
)
from agentic_rag.ingestion.watcher import FileState, diff_snapshots, snapshot
```
- `from __future__ import annotations` — defers type hint evaluation for modern syntax.
- `from dataclasses import dataclass` — imports the decorator for the `SyncResult` type below.
- `from pathlib import Path` — used to type the watched folder argument.
- `from agentic_rag.ingestion.pipeline import IngestedDocument, IngestionFailure, process_changes` — imports the two result types and the function that does the actual per-file conversion/chunking/tagging/validation work (see `ingestion/pipeline.py`).
- `from agentic_rag.ingestion.watcher import FileState, diff_snapshots, snapshot` — imports the fingerprint type and the two functions that produce a fresh snapshot of the folder and compare it against a previous one (see `ingestion/watcher.py`). Note the local parameter name `snapshot` (used later in this file, e.g. the function argument `previous_snapshot`) and the imported function `snapshot` share a name in spirit — Python resolves this fine because they're in different scopes, but readers should note `snapshot(...)` called inside this file always refers to the imported watcher function, not any local variable.

### Lines 14-19 — `SyncResult`
```python
@dataclass(frozen=True)
class SyncResult:
    current_snapshot: dict[str, FileState]
    documents: list[IngestedDocument]
    failures: list[IngestionFailure]
    deleted: list[str]
```
- `@dataclass(frozen=True)` — an immutable result record, consistent with the rest of the ingestion module's pattern of using frozen dataclasses for "this is a finished fact" values.
- `current_snapshot: dict[str, FileState]` — the freshly-taken fingerprint of the folder's current state, meant to become the `previous_snapshot` for the *next* sync cycle.
- `documents: list[IngestedDocument]` — every file that was successfully converted/chunked/tagged/validated this cycle (i.e. every created or modified file that processed cleanly).
- `failures: list[IngestionFailure]` — every file that failed processing this cycle, with a reason.
- `deleted: list[str]` — the relative paths of files that were removed from disk since the last snapshot; this list is exactly what needs to be removed from the search index, though this file itself doesn't perform that removal.

### Lines 22-27 — `sync_folder` function signature and docstring
```python
def sync_folder(
    folder: Path,
    previous_snapshot: dict[str, FileState],
    chunk_size_chars: int,
    known_tiers: list[str],
) -> SyncResult:
    """Run one ingestion cycle: detect what changed since `previous_snapshot`
    and convert/chunk/tag it.

    This is the one place edit and delete propagation (FR4) originates from:
    the returned `deleted` paths and freshly (re)ingested `documents` are
    exactly what the indexing phase needs to keep Qdrant in sync with the
    watched folder. How often this runs is a scheduling concern for later
    (Phase 7); this function only answers "what changed since last time."
    """
```
- `def sync_folder(folder: Path, previous_snapshot: dict[str, FileState], chunk_size_chars: int, known_tiers: list[str]) -> SyncResult:` — takes the watched folder's root path, the previous cycle's snapshot to diff against, the target chunk size, and the list of valid access tiers; returns a `SyncResult` describing everything that changed and how it processed.
- The docstring frames this function's role in the bigger system: it is the origin point of "edit and delete propagation" (referred to by the requirement tag FR4 in the project's requirements doc) — i.e., the mechanism by which the search index eventually stays in sync with the actual state of the watched folder. It's explicit that scheduling *when* and *how often* this runs is a separate concern handled elsewhere (`ingestion/scheduler.py`); this function is only about correctly answering "what changed" for one point-in-time comparison.

### Lines 37-41 — Detecting changes and processing them
```python
    current_snapshot = snapshot(folder)
    changes = diff_snapshots(previous_snapshot, current_snapshot)
    documents, failures = process_changes(
        folder, changes, chunk_size_chars, known_tiers
    )
```
- `current_snapshot = snapshot(folder)` — takes a fresh fingerprint of every file currently in the watched folder, calling `watcher.py`'s `snapshot()` function.
- `changes = diff_snapshots(previous_snapshot, current_snapshot)` — compares that fresh fingerprint against the snapshot passed in from the previous cycle, producing a `FolderChanges` object listing created, modified, and deleted paths.
- `documents, failures = process_changes(folder, changes, chunk_size_chars, known_tiers)` — hands the created/modified files off to `pipeline.py`'s `process_changes()`, which converts, chunks, tags, and validates each one, returning the successfully processed documents and any per-file failures (deletions aren't touched here, since there's nothing to convert for a file that no longer exists).

### Lines 43-48 — Returning the composed result
```python
    return SyncResult(
        current_snapshot=current_snapshot,
        documents=documents,
        failures=failures,
        deleted=changes.deleted,
    )
```
- Packages everything computed in this cycle — the fresh snapshot (to be persisted and used as next cycle's starting point), the successfully processed documents, the processing failures, and the list of deleted file paths (pulled straight from `changes.deleted`) — into a single `SyncResult` and returns it to the caller. The caller (`ingestion/scheduler.py`'s `run_sync_cycle()`) is the piece that actually takes `documents` and pushes them into the Qdrant index, and takes `deleted` and removes those entries from the index.
