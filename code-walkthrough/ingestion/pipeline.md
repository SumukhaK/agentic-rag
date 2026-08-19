# `ingestion/pipeline.py`

**Purpose:** This file is the orchestrator that turns a batch of "files that changed on disk" into fully-processed, ready-to-index documents. For every created or modified file, it runs the full chain of steps — figure out its access tier from its folder, convert it to Markdown, split it into chunks, and validate the result — and it does this per-file, in isolation, so that one broken or unsupported file (a corrupt PDF, an unrecognized folder, an empty document) can't halt processing for every other file in the same batch. This matters because this pipeline is meant to be called repeatedly, forever, by a background sync job (see `ingestion/scheduler.py`), and a single bad document must never be able to permanently wedge the whole ingestion process.

## Line-by-line walkthrough

### Lines 1-10 — Imports
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.chunker import Chunk, chunk_markdown
from agentic_rag.ingestion.converter import convert_to_markdown
from agentic_rag.ingestion.tagger import access_tier_for
from agentic_rag.ingestion.validation import validate_document
from agentic_rag.ingestion.watcher import FolderChanges
```
- `from __future__ import annotations` — defers evaluation of type hints so modern syntax (like `list[Chunk]`) works smoothly.
- `from dataclasses import dataclass` — imports the decorator used to define the two simple result types below.
- `from pathlib import Path` — used to type the folder argument and to join folder + relative path when locating a file on disk.
- `from agentic_rag.ingestion.chunker import Chunk, chunk_markdown` — brings in the `Chunk` type and the function that splits Markdown text into chunks (see `ingestion/chunker.py`).
- `from agentic_rag.ingestion.converter import convert_to_markdown` — brings in the function that turns any source file into Markdown text (see `ingestion/converter.py`).
- `from agentic_rag.ingestion.tagger import access_tier_for` — brings in the function that derives a document's access tier (e.g. "manager", "employee") from its folder path (see `ingestion/tagger.py`).
- `from agentic_rag.ingestion.validation import validate_document` — brings in the function that checks a processed document is actually usable before it's allowed through (see `ingestion/validation.py`).
- `from agentic_rag.ingestion.watcher import FolderChanges` — imports the type describing which files were created/modified/deleted since the last check (produced by `ingestion/watcher.py`); this pipeline consumes that type as input.

### Lines 13-18 — The `IngestedDocument` result type
```python
@dataclass(frozen=True)
class IngestedDocument:
    relative_path: str
    markdown: str
    chunks: list[Chunk]
    access_tier: str
```
- `@dataclass(frozen=True)` — an immutable data container, same reasoning as `Chunk` in the chunker: once a document has finished processing, nothing should mutate its fields.
- `relative_path: str` — the file's path relative to the watched folder root, used as its stable identity throughout the system (for indexing, deletion, and re-sync matching).
- `markdown: str` — the file's full converted Markdown text (kept around, e.g. useful for debugging or re-processing without re-converting).
- `chunks: list[Chunk]` — the list of `Chunk` objects produced from that Markdown, ready to be embedded and stored.
- `access_tier: str` — which access tier (e.g. "manager", "employee") this document belongs to, used later to filter search results by the querying user's permission level.

### Lines 21-24 — The `IngestionFailure` result type
```python
@dataclass(frozen=True)
class IngestionFailure:
    relative_path: str
    reason: str
```
- `@dataclass(frozen=True)` — again, an immutable record.
- `relative_path: str` — identifies which file failed to process.
- `reason: str` — a human-readable description of why it failed (built from the caught exception's type and message), so a caller/operator can see what went wrong without needing a full stack trace.

### Lines 27-47 — `process_changes` function signature and docstring
```python
def process_changes(
    folder: Path,
    changes: FolderChanges,
    chunk_size_chars: int,
    known_tiers: list[str],
) -> tuple[list[IngestedDocument], list[IngestionFailure]]:
    """Convert, chunk, access-tag, and validate every created/modified file
    in `changes`.

    A file that fails at any step - unrecognized access tier, a conversion
    error, failing schema validation (e.g. zero usable chunks), or anything
    else - doesn't abort the rest of the batch. It's reported as an
    IngestionFailure alongside the IngestedDocuments for every other file
    that succeeded. This function is the one Phase 7's scheduled sync job
    will call repeatedly: a single corrupted, unsupported, or blank file
    must not be able to permanently stall every other document in the
    corpus by raising on every run.

    Deletions carry nothing to convert; propagating them to the index is
    the indexing phase's responsibility, not this pipeline step's.
    """
```
- `def process_changes(folder: Path, changes: FolderChanges, chunk_size_chars: int, known_tiers: list[str]) -> tuple[list[IngestedDocument], list[IngestionFailure]]:` — the function's signature: it takes the watched folder's root path, the set of detected changes (from the watcher), the target chunk size in characters, and the list of valid access-tier names; it returns a pair of lists — successfully processed documents, and failures.
- The docstring lays out the resilience contract explained in the Purpose section above: any failure at any stage becomes an `IngestionFailure` entry rather than an exception that kills the whole batch, because this function is called over and over by a long-running background sync job and one bad file must never be a permanent outage for everything else.
- It also clarifies scope: `changes.deleted` (files removed from disk) isn't touched here at all — there's nothing to convert for a deleted file — so propagating deletions to the search index is left to the indexing layer, keeping this function focused purely on files that need (re-)processing.

### Lines 48-49 — Result accumulators
```python
    documents: list[IngestedDocument] = []
    failures: list[IngestionFailure] = []
```
- `documents` — will collect every file that made it through the full pipeline successfully.
- `failures` — will collect every file that failed at any step, along with why.

### Lines 51-52 — Iterating over changed files
```python
    for relative_path in changes.created + changes.modified:
        try:
```
- `changes.created + changes.modified` — concatenates the "newly created" and "modified since last check" file lists into one list to process. Both cases need the exact same treatment (convert, chunk, tag, validate from scratch) — a modified file doesn't get any special incremental handling, it's just fully reprocessed.
- `try:` — opens a per-file try block; this is the mechanism that isolates one file's failure from the rest of the loop.

### Lines 53-62 — The processing chain for one file
```python
            access_tier = access_tier_for(relative_path, known_tiers)
            markdown = convert_to_markdown(folder / relative_path)
            chunks = chunk_markdown(markdown, chunk_size_chars)
            document = IngestedDocument(
                relative_path=relative_path,
                markdown=markdown,
                chunks=chunks,
                access_tier=access_tier,
            )
            validate_document(document)
```
- `access_tier = access_tier_for(relative_path, known_tiers)` — first, derives which access tier this file belongs to from its folder path. This is done before conversion (which is more expensive) so that an untagged or misfiled document fails fast, without wasting time converting it.
- `markdown = convert_to_markdown(folder / relative_path)` — joins the watched folder's root with the file's relative path to get its full location on disk, then converts it to Markdown text via `markitdown` (through `ingestion/converter.py`).
- `chunks = chunk_markdown(markdown, chunk_size_chars)` — splits the converted Markdown into a list of `Chunk` objects sized around `chunk_size_chars`.
- `document = IngestedDocument(...)` — assembles the finished `IngestedDocument` record from everything computed so far.
- `validate_document(document)` — runs the final quality gate (see `ingestion/validation.py`), which raises an exception if, for example, the document produced zero usable chunks or is missing its access tier — turning a silently-broken document into a loud, reported failure instead of letting it quietly enter the index.

### Lines 63-68 — Catching and recording any failure
```python
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append(
                IngestionFailure(relative_path=relative_path, reason=reason)
            )
            continue
```
- `except Exception as exc:` — catches literally any exception raised anywhere in the chain above (an unrecognized tier, a conversion error from `markitdown`, a validation failure, or anything unexpected) — deliberately broad, per the docstring's resilience guarantee that no single file's problem should be able to escape and kill the loop.
- `reason = f"{type(exc).__name__}: {exc}"` — builds a readable string combining the exception's class name (e.g. `UnknownAccessTierError`) with its message, so the failure record is self-explanatory without needing the original traceback.
- `failures.append(IngestionFailure(relative_path=relative_path, reason=reason))` — records the failure for this specific file.
- `continue` — skips the rest of the loop body (the success-path `documents.append` below) and moves on to the next file in the batch.

### Lines 70-72 — Recording success and returning
```python
        documents.append(document)

    return documents, failures
```
- `documents.append(document)` — this line only runs if the entire `try` block completed without raising, meaning the file was successfully converted, chunked, tagged, and validated; it's added to the successful-documents list.
- `return documents, failures` — after the loop has processed every changed file, returns both lists together as a tuple, giving the caller (`ingestion/sync.py`) a complete picture of what succeeded and what didn't in this batch.
