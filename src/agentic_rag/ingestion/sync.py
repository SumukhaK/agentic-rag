from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.pipeline import (
    IngestedDocument,
    IngestionFailure,
    process_changes,
)
from agentic_rag.ingestion.watcher import FileState, diff_snapshots, snapshot


@dataclass(frozen=True)
class SyncResult:
    snapshot: dict[str, FileState]
    documents: list[IngestedDocument]
    failures: list[IngestionFailure]
    deleted: list[str]


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
    current_snapshot = snapshot(folder)
    changes = diff_snapshots(previous_snapshot, current_snapshot)
    documents, failures = process_changes(
        folder, changes, chunk_size_chars, known_tiers
    )

    return SyncResult(
        snapshot=current_snapshot,
        documents=documents,
        failures=failures,
        deleted=changes.deleted,
    )
