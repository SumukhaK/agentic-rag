from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.chunker import Chunk, chunk_markdown
from agentic_rag.ingestion.converter import convert_to_markdown
from agentic_rag.ingestion.tagger import (
    UnknownAccessTierError,
    UntaggedDocumentError,
    access_tier_for,
)
from agentic_rag.ingestion.watcher import FolderChanges


@dataclass(frozen=True)
class IngestedDocument:
    relative_path: str
    markdown: str
    chunks: list[Chunk]
    access_tier: str


@dataclass(frozen=True)
class IngestionFailure:
    relative_path: str
    reason: str


def process_changes(
    folder: Path,
    changes: FolderChanges,
    chunk_size_chars: int,
    known_tiers: list[str],
) -> tuple[list[IngestedDocument], list[IngestionFailure]]:
    """Convert, chunk, and access-tag every created/modified file in `changes`.

    A file whose access tier can't be determined doesn't abort the rest of
    the batch - it's reported as an IngestionFailure alongside the
    IngestedDocuments for every other, validly-tagged file. Access control
    correctness matters, but one misplaced file shouldn't block ingestion of
    everything else in the same watcher cycle.

    Deletions carry nothing to convert; propagating them to the index is
    the indexing phase's responsibility, not this pipeline step's.
    """
    documents: list[IngestedDocument] = []
    failures: list[IngestionFailure] = []

    for relative_path in changes.created + changes.modified:
        try:
            access_tier = access_tier_for(relative_path, known_tiers)
        except (UntaggedDocumentError, UnknownAccessTierError) as exc:
            failures.append(
                IngestionFailure(relative_path=relative_path, reason=str(exc))
            )
            continue

        markdown = convert_to_markdown(folder / relative_path)
        chunks = chunk_markdown(markdown, chunk_size_chars)
        documents.append(
            IngestedDocument(
                relative_path=relative_path,
                markdown=markdown,
                chunks=chunks,
                access_tier=access_tier,
            )
        )

    return documents, failures
