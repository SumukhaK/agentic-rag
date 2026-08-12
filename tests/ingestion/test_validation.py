import pytest

from agentic_rag.ingestion.chunker import Chunk
from agentic_rag.ingestion.pipeline import IngestedDocument
from agentic_rag.ingestion.validation import DocumentValidationError, validate_document


def _document(**overrides):
    defaults = dict(
        relative_path="tier-1/a.txt",
        markdown="Arsenal drew 1-1.",
        chunks=[Chunk(text="Arsenal drew 1-1.", index=0)],
        access_tier="tier-1",
    )
    defaults.update(overrides)
    return IngestedDocument(**defaults)


def test_validate_document_accepts_a_well_formed_document():
    validate_document(_document())  # should not raise


def test_validate_document_raises_for_a_document_with_no_chunks():
    with pytest.raises(DocumentValidationError):
        validate_document(_document(chunks=[]))


def test_validate_document_raises_for_a_chunk_with_empty_text():
    with pytest.raises(DocumentValidationError):
        validate_document(_document(chunks=[Chunk(text="   ", index=0)]))


def test_validate_document_raises_for_a_document_with_no_access_tier():
    with pytest.raises(DocumentValidationError):
        validate_document(_document(access_tier=""))
