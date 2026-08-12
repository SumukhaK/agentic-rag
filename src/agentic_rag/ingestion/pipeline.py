from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.chunker import Chunk, chunk_markdown
from agentic_rag.ingestion.converter import convert_to_markdown
from agentic_rag.ingestion.tagger import access_tier_for
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

    A file that fails at any step - unrecognized access tier, a conversion
    error, or anything else - doesn't abort the rest of the batch. It's
    reported as an IngestionFailure alongside the IngestedDocuments for
    every other file that succeeded. This function is the one Phase 7's
    scheduled sync job will call repeatedly: a single corrupted or
    unsupported file must not be able to permanently stall every other
    document in the corpus by raising on every run.

    Deletions carry nothing to convert; propagating them to the index is
    the indexing phase's responsibility, not this pipeline step's.
    """
    documents: list[IngestedDocument] = []
    failures: list[IngestionFailure] = []

    for relative_path in changes.created + changes.modified:
        try:
            access_tier = access_tier_for(relative_path, known_tiers)
            markdown = convert_to_markdown(folder / relative_path)
            chunks = chunk_markdown(markdown, chunk_size_chars)
        except Exception as exc:
            failures.append(
                IngestionFailure(relative_path=relative_path, reason=str(exc))
            )
            continue

        documents.append(
            IngestedDocument(
                relative_path=relative_path,
                markdown=markdown,
                chunks=chunks,
                access_tier=access_tier,
            )
        )

    return documents, failures
