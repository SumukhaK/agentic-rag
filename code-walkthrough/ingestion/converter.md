# `ingestion/converter.py`

**Purpose:** This tiny file is the "front door" of ingestion: it takes a source file on disk (which could be a PDF, Word document, PowerPoint, plain text file, or many other formats) and converts it into plain Markdown text. Everything downstream in the ingestion pipeline (chunking, tagging, validation, embedding) is written to work on Markdown text, not on a dozen different file formats — so this converter's job is to be the single place where format-specific complexity is absorbed, using the third-party `markitdown` library to do the actual format parsing.

## Line-by-line walkthrough

### Lines 1-5 — Imports and module-level converter instance
```python
from pathlib import Path

from markitdown import MarkItDown

_converter = MarkItDown()
```
- `from pathlib import Path` — imports Python's modern, object-oriented filesystem path type, used for the function's input parameter so callers pass a `Path` object rather than a raw string.
- `from markitdown import MarkItDown` — imports the `MarkItDown` class from the third-party `markitdown` package (a Microsoft library that converts many document formats — PDF, DOCX, PPTX, XLSX, HTML, images with OCR, etc. — into Markdown text).
- `_converter = MarkItDown()` — creates a single, module-level instance of the converter at import time, rather than creating a new one inside the function on every call. The leading underscore marks it as a private module variable. Reusing one instance avoids repeatedly paying whatever setup cost `MarkItDown()` has (e.g. initializing internal format handlers) on every single file conversion.

### Lines 8-11 — The `convert_to_markdown` function
```python
def convert_to_markdown(path: Path) -> str:
    """Convert any supported source file to Markdown text via markitdown."""
    result = _converter.convert(str(path))
    return result.text_content
```
- `def convert_to_markdown(path: Path) -> str:` — the public function other ingestion modules (notably `ingestion/pipeline.py`) call. It takes the path to a source file and returns the converted Markdown as a plain string.
- The docstring is a one-line summary: this function's whole job is delegating to `markitdown`, whatever the underlying file type is.
- `result = _converter.convert(str(path))` — calls `markitdown`'s `convert` method, which expects a string path (hence `str(path)` to convert from a `Path` object), and returns a result object containing the conversion output. `markitdown` internally detects the file type (by extension/content) and picks the right internal parser, so this file itself never needs to branch on file type.
- `return result.text_content` — extracts just the Markdown text from `markitdown`'s result object (which may carry other metadata too) and returns it to the caller. Any errors during conversion (e.g. an unsupported or corrupted file) simply propagate up as exceptions — this function doesn't catch them, leaving error handling to the caller (`ingestion/pipeline.py`, which wraps calls to this function in a `try`/`except` so one bad file doesn't stop the rest of the batch).
