from agentic_rag.ingestion.chunker import chunk_markdown


def test_chunk_markdown_packs_small_blocks_into_one_chunk():
    markdown = "First paragraph.\n\nSecond paragraph."

    chunks = chunk_markdown(markdown, chunk_size_chars=1000)

    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."
    assert chunks[0].index == 0


def test_chunk_markdown_starts_new_chunk_when_limit_would_be_exceeded():
    block_a = "A" * 30
    block_b = "B" * 30
    markdown = f"{block_a}\n\n{block_b}"

    chunks = chunk_markdown(markdown, chunk_size_chars=40)

    assert [c.text for c in chunks] == [block_a, block_b]
    assert [c.index for c in chunks] == [0, 1]


def test_chunk_markdown_keeps_oversized_block_intact_without_splitting():
    huge_table_block = "\n".join("| a | b |" for _ in range(50))  # one block, over the limit
    markdown = f"intro\n\n{huge_table_block}\n\noutro"

    chunks = chunk_markdown(markdown, chunk_size_chars=100)

    # the oversized block is never split mid-way, even though it exceeds
    # chunk_size_chars — that's the "hybrid" part of hybrid chunking.
    assert [c.text for c in chunks] == ["intro", huge_table_block, "outro"]


def test_chunk_markdown_returns_empty_list_for_blank_input():
    assert chunk_markdown("   \n\n  ", chunk_size_chars=1000) == []


def test_chunk_markdown_assigns_sequential_indices():
    markdown = "\n\n".join(f"block {i}" for i in range(5))

    chunks = chunk_markdown(markdown, chunk_size_chars=15)

    assert [c.index for c in chunks] == list(range(len(chunks)))
