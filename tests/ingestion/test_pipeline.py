from agentic_rag.ingestion.pipeline import process_changes
from agentic_rag.ingestion.watcher import FolderChanges


def test_process_changes_converts_created_and_modified_files(tmp_path):
    (tmp_path / "a.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / "b.txt").write_text("Chelsea won 3-0.")
    changes = FolderChanges(created=["a.txt"], modified=["b.txt"], deleted=[])

    documents = process_changes(tmp_path, changes, chunk_size_chars=2000)

    assert [doc.relative_path for doc in documents] == ["a.txt", "b.txt"]
    assert "Arsenal drew 1-1." in documents[0].markdown
    assert "Chelsea won 3-0." in documents[1].markdown


def test_process_changes_chunks_each_document(tmp_path):
    (tmp_path / "a.txt").write_text("Arsenal drew 1-1.")
    changes = FolderChanges(created=["a.txt"], modified=[], deleted=[])

    documents = process_changes(tmp_path, changes, chunk_size_chars=2000)

    assert len(documents[0].chunks) == 1
    assert documents[0].chunks[0].text == "Arsenal drew 1-1."
    assert documents[0].chunks[0].index == 0


def test_process_changes_respects_chunk_size_chars(tmp_path):
    block_a = "A" * 30
    block_b = "B" * 30
    (tmp_path / "a.txt").write_text(f"{block_a}\n\n{block_b}")
    changes = FolderChanges(created=["a.txt"], modified=[], deleted=[])

    documents = process_changes(tmp_path, changes, chunk_size_chars=40)

    assert [c.text for c in documents[0].chunks] == [block_a, block_b]


def test_process_changes_ignores_deleted_files(tmp_path):
    changes = FolderChanges(created=[], modified=[], deleted=["c.txt"])

    documents = process_changes(tmp_path, changes, chunk_size_chars=2000)

    assert documents == []


def test_process_changes_returns_empty_list_for_no_changes(tmp_path):
    changes = FolderChanges(created=[], modified=[], deleted=[])

    documents = process_changes(tmp_path, changes, chunk_size_chars=2000)

    assert documents == []
