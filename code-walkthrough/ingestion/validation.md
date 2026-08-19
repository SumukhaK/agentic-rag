# `ingestion/validation.py`

**Purpose:** This file is the final quality gate a processed document must pass through before it's allowed to be indexed and searchable. Even if a file converted and chunked without raising any errors, the *result* could still be useless (e.g. a document that produced zero chunks, or a chunk that's just blank whitespace, or a document that somehow ended up with no access tier). Per the project's data-quality philosophy, these problems must be loud, explicit failures rather than something that silently slips through and pollutes the search index with junk. This module defines exactly what "acceptable" means for a processed document and enforces it.

## Line-by-line walkthrough

### Lines 1-6 — Imports and type-checking-only import
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.ingestion.pipeline import IngestedDocument
```
- `from __future__ import annotations` — defers evaluation of type hints, allowing the string-quoted forward reference used below and modern hint syntax to work cleanly.
- `from typing import TYPE_CHECKING` — imports a constant that is always `False` at runtime but treated as `True` by static type checkers (like `mypy`).
- `if TYPE_CHECKING: from agentic_rag.ingestion.pipeline import IngestedDocument` — imports the `IngestedDocument` type only for type-checking purposes, not at actual runtime. This is a deliberate way to avoid a circular import: `pipeline.py` imports from `validation.py` (to call `validate_document`), so if `validation.py` imported `IngestedDocument` from `pipeline.py` normally at the top level, the two modules would import each other in a loop. Guarding the import behind `TYPE_CHECKING` means type checkers still see the real type for hints, but at actual runtime this line never executes, so there's no circular-import problem.

### Lines 9-11 — `DocumentValidationError`
```python
class DocumentValidationError(Exception):
    """Raised when a processed document doesn't meet the invariants required
    before it's handed to the indexing phase."""
```
- Defines a dedicated exception type for validation failures specifically, rather than reusing a generic exception — this lets the failure be identified as "this document failed our quality checks" as opposed to, say, a conversion crash, if any caller ever wants to distinguish the two.

### Lines 14-22 — `validate_document` function signature and docstring
```python
def validate_document(document: "IngestedDocument") -> None:
    """Check the invariants an IngestedDocument must hold before indexing.

    IngestedDocument/Chunk (pipeline.py, chunker.py) define the schema's
    shape; this is the validation step data-quality failures must not
    silently pass through - it's what makes a bad document (e.g. one that
    converted to zero usable chunks) a loud, reported failure instead of a
    document quietly entering the index with nothing useful in it.
    """
```
- `def validate_document(document: "IngestedDocument") -> None:` — the function's signature. The type hint is written as the string `"IngestedDocument"` (a forward reference) rather than the bare class name, because — as explained above — the real class isn't actually imported at runtime, only during type checking. The function returns `None`; it communicates failure exclusively by raising, not by returning a boolean or error code, which is what makes a bad document impossible to silently ignore (a raised exception forces the caller to either handle it or let it propagate — it can't be accidentally left unchecked the way a returned `False` could be).
- The docstring clarifies the division of responsibility: `pipeline.py` and `chunker.py` define what shape a document/chunk should have (their dataclass fields), while this function is what actually *enforces* that the content within that shape is meaningful, not just structurally present.

### Lines 23-26 — Checking for at least one chunk
```python
    if not document.chunks:
        raise DocumentValidationError(
            f"'{document.relative_path}' produced no chunks"
        )
```
- `if not document.chunks:` — checks whether the document's `chunks` list is empty. This can happen, for example, if the source file converted to Markdown but the Markdown text was itself empty or whitespace-only, so `chunk_markdown()` returned no chunks.
- `raise DocumentValidationError(f"'{document.relative_path}' produced no chunks")` — if so, raises with a message identifying exactly which file is at fault, so the failure is traceable back to its source.

### Lines 28-32 — Checking every chunk has real content
```python
    for chunk in document.chunks:
        if not chunk.text.strip():
            raise DocumentValidationError(
                f"'{document.relative_path}' has an empty chunk at index {chunk.index}"
            )
```
- `for chunk in document.chunks:` — iterates through every chunk in the document (not just checking that the list is non-empty, but that each individual entry is meaningful).
- `if not chunk.text.strip():` — checks whether a given chunk's text, once whitespace is trimmed, is actually empty. A chunk could technically exist (a non-empty entry in the list) yet contain only blank lines or spaces, which would be useless once embedded and indexed.
- `raise DocumentValidationError(f"'{document.relative_path}' has an empty chunk at index {chunk.index}")` — raises with both the document's path and the specific chunk's index, pinpointing exactly which chunk is the problem.

### Lines 34-37 — Checking the access tier is set
```python
    if not document.access_tier:
        raise DocumentValidationError(
            f"'{document.relative_path}' has no access tier"
        )
```
- `if not document.access_tier:` — checks that the document's `access_tier` field is a non-empty string. In normal operation this should already be guaranteed by `tagger.py`'s `access_tier_for()` (which either returns a valid tier or raises before an `IngestedDocument` is even constructed), so this acts as a defensive backstop invariant check rather than something expected to trigger in practice — it protects against the field ever ending up blank via some other code path in the future.
- `raise DocumentValidationError(f"'{document.relative_path}' has no access tier")` — raises, naming the affected document. If none of the three checks above trigger, the function simply returns `None` implicitly, meaning the document is considered valid and safe to index.
