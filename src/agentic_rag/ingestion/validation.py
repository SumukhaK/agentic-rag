from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.ingestion.pipeline import IngestedDocument


class DocumentValidationError(Exception):
    """Raised when a processed document doesn't meet the invariants required
    before it's handed to the indexing phase."""


def validate_document(document: "IngestedDocument") -> None:
    """Check the invariants an IngestedDocument must hold before indexing.

    IngestedDocument/Chunk (pipeline.py, chunker.py) define the schema's
    shape; this is the validation step data-quality failures must not
    silently pass through - it's what makes a bad document (e.g. one that
    converted to zero usable chunks) a loud, reported failure instead of a
    document quietly entering the index with nothing useful in it.
    """
    if not document.chunks:
        raise DocumentValidationError(
            f"'{document.relative_path}' produced no chunks"
        )

    for chunk in document.chunks:
        if not chunk.text.strip():
            raise DocumentValidationError(
                f"'{document.relative_path}' has an empty chunk at index {chunk.index}"
            )

    if not document.access_tier:
        raise DocumentValidationError(
            f"'{document.relative_path}' has no access tier"
        )
