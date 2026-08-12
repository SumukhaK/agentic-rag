from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.converter import convert_to_markdown
from agentic_rag.ingestion.watcher import FolderChanges


@dataclass(frozen=True)
class IngestedDocument:
    relative_path: str
    markdown: str


def process_changes(folder: Path, changes: FolderChanges) -> list[IngestedDocument]:
    """Convert every created/modified file in `changes` to Markdown.

    Deletions carry nothing to convert; propagating them to the index is
    the indexing phase's responsibility, not this pipeline step's.
    """
    documents = []
    for relative_path in changes.created + changes.modified:
        markdown = convert_to_markdown(folder / relative_path)
        documents.append(
            IngestedDocument(relative_path=relative_path, markdown=markdown)
        )
    return documents
