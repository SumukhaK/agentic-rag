from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK_SEPARATOR = re.compile(r"\n{2,}")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


def chunk_markdown(markdown: str, chunk_size_chars: int) -> list[Chunk]:
    """Split Markdown into chunks of roughly `chunk_size_chars`.

    Chunking is block-based (blocks are separated by a blank line, which in
    Markdown groups a paragraph, list, or table as one unit): blocks are
    packed into a chunk up to the target size, and a chunk is closed out as
    soon as the next block would exceed it. A single block that is itself
    larger than `chunk_size_chars` (e.g. a big table) is never split — it
    becomes its own, oversized chunk rather than losing context mid-block.
    """
    blocks = [b.strip() for b in _BLOCK_SEPARATOR.split(markdown.strip()) if b.strip()]

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

    return [Chunk(text=text, index=i) for i, text in enumerate(chunks)]
