# `ingestion/chunker.py`

**Purpose:** This file takes a document that has already been converted into Markdown text and splits it into smaller pieces ("chunks") that are small enough to be embedded and stored in the vector database, but large enough to still contain useful context. Chunking matters a lot for retrieval-augmented generation (RAG): if chunks are too big, the embedding blurs together too many unrelated ideas and retrieval gets less precise; if chunks are too small (or cut mid-sentence/mid-table), the assistant loses context and may misread a fact. This file's strategy is to chunk along natural document boundaries (blank-line-separated blocks like paragraphs, list items, or tables) rather than cutting text at an arbitrary character count.

## Line-by-line walkthrough

### Lines 1-6 — Imports and module-level setup
```python
from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK_SEPARATOR = re.compile(r"\n{2,}")
```
- `from __future__ import annotations` — makes all type hints in this file lazily evaluated as strings, which lets the code use modern type-hint syntax (like `list[Chunk]`) without worrying about Python version compatibility for evaluating those hints at import time.
- `import re` — brings in Python's regular-expression module, needed to detect where one Markdown "block" ends and another begins.
- `from dataclasses import dataclass` — imports the `@dataclass` decorator, used below to define a simple, boilerplate-free data container class.
- `_BLOCK_SEPARATOR = re.compile(r"\n{2,}")` — pre-compiles (for efficiency, since it will be reused on every call) a regex that matches two or more consecutive newline characters. In Markdown, a blank line (i.e., two or more newlines in a row) is what separates one paragraph/list/table "block" from the next, so this pattern is the tool used to break the document into those blocks. The leading underscore signals it's a private, module-internal constant not meant to be imported by other files.

### Lines 9-12 — The `Chunk` data class
```python
@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
```
- `@dataclass(frozen=True)` — declares `Chunk` as an immutable data class: Python auto-generates `__init__`, `__eq__`, and `__repr__` for it, and `frozen=True` means once a `Chunk` is created, its fields can't be reassigned. Immutability here is a safety choice — a chunk represents a finished, already-decided piece of text, so nothing downstream should be able to accidentally mutate it.
- `text: str` — the actual chunk's Markdown content.
- `index: int` — the chunk's position (0-based) within the document it came from, so downstream code (like the indexer) can keep chunks in order and refer back to "chunk 3 of document X."

### Lines 15-24 — `chunk_markdown` function signature and docstring
```python
def chunk_markdown(markdown: str, chunk_size_chars: int) -> list[Chunk]:
    """Split Markdown into chunks of roughly `chunk_size_chars`.

    Chunking is block-based (blocks are separated by a blank line, which in
    Markdown groups a paragraph, list, or table as one unit): blocks are
    packed into a chunk up to the target size, and a chunk is closed out as
    soon as the next block would exceed it. A single block that is itself
    larger than `chunk_size_chars` (e.g. a big table) is never split — it
    becomes its own, oversized chunk rather than losing context mid-block.
    """
```
- `def chunk_markdown(markdown: str, chunk_size_chars: int) -> list[Chunk]:` — the function's public entry point. It takes the full Markdown text of a document and a target chunk size (measured in characters), and returns a list of `Chunk` objects.
- The docstring explains the design decision at the heart of this file: rather than blindly cutting every `chunk_size_chars` characters (which could slice a sentence, list item, or table row in half), it packs whole blocks together up to roughly the target size, and — critically — never splits a single block even if that block alone is bigger than the target. This trades strict size uniformity for never destroying the meaning of a block, which matters more for retrieval quality than exact chunk-size consistency.

### Line 25 — Splitting the document into blocks
```python
    blocks = [b.strip() for b in _BLOCK_SEPARATOR.split(markdown.strip()) if b.strip()]
```
- `markdown.strip()` — first removes leading/trailing whitespace from the whole document so an accidental blank line at the very start or end doesn't create an empty phantom block.
- `_BLOCK_SEPARATOR.split(...)` — uses the earlier-compiled regex to break the text everywhere there are two-or-more consecutive newlines, producing a list of raw block strings (paragraphs, list groups, tables, etc.).
- `for b in ... if b.strip()` — filters out any block that is empty or whitespace-only (which can happen if there were three or more blank lines in a row, producing an empty string between separators).
- `b.strip()` (in the list comprehension body) — trims each individual block's own leading/trailing whitespace, so blocks are clean before being measured and joined.
- Overall this line produces `blocks`, a list of non-empty, trimmed Markdown blocks in their original document order.

### Lines 27-38 — Packing blocks into chunks
```python
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if not current:
            current = block
        elif len(current) + 2 + len(block) <= chunk_size_chars:
            current = f"{current}\n\n{block}"
        else:
            chunks.append(current)
            current = block
    if current:
        chunks.append(current)
```
- `chunks: list[str] = []` — will accumulate the finished chunk strings.
- `current = ""` — the chunk currently being built up, starting empty.
- `for block in blocks:` — iterates through every block in document order, deciding for each one whether it can be added to the chunk-in-progress or whether it needs to start a new chunk.
- `if not current:` — if nothing has been added to the current chunk yet (i.e., this is the very first block, or the previous chunk was just closed out), the block simply becomes the start of the new current chunk. This is also the path that lets an oversized single block become its own chunk — it's placed into `current` unconditionally here regardless of its size.
- `elif len(current) + 2 + len(block) <= chunk_size_chars:` — otherwise, checks whether appending this block (plus the 2 characters for the `\n\n` separator that will rejoin them) would still fit within the target `chunk_size_chars`. If so, it's safe to keep packing.
- `current = f"{current}\n\n{block}"` — appends the block to the current chunk, rejoined with a blank line so the Markdown structure (block separation) is preserved inside the chunk's own text.
- `else:` — if adding the block would exceed the target size, the current chunk is considered "full."
- `chunks.append(current)` — the completed chunk is saved to the results list...
- `current = block` — ...and a new chunk is started with this block as its first content (mirroring the `if not current` branch, but here it's because the *previous* chunk was closed, not because there was no chunk yet).
- `if current: chunks.append(current)` — after the loop ends, whatever is left in `current` (the last chunk being built) hasn't been appended yet, so this final check flushes it into `chunks`. The `if current` guard avoids appending an empty string in the edge case where `blocks` was empty (e.g. an empty document).

### Line 40 — Building the returned `Chunk` objects
```python
    return [Chunk(text=text, index=i) for i, text in enumerate(chunks)]
```
- Converts the plain list of chunk strings into a list of `Chunk` dataclass instances, using `enumerate` to assign each one its 0-based `index` in document order. This is the final return value handed back to callers like `ingestion/pipeline.py`.
