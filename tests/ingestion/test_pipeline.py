from agentic_rag.ingestion.pipeline import process_changes
from agentic_rag.ingestion.watcher import FolderChanges

KNOWN_TIERS = ["tier-1", "tier-2", "tier-3"]


def test_process_changes_converts_created_and_modified_files(tmp_path):
    (tmp_path / "tier-1").mkdir()
    (tmp_path / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / "tier-2").mkdir()
    (tmp_path / "tier-2" / "b.txt").write_text("Chelsea won 3-0.")
    changes = FolderChanges(
        created=["tier-1/a.txt"], modified=["tier-2/b.txt"], deleted=[]
    )

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert failures == []
    assert [doc.relative_path for doc in documents] == ["tier-1/a.txt", "tier-2/b.txt"]
    assert "Arsenal drew 1-1." in documents[0].markdown
    assert "Chelsea won 3-0." in documents[1].markdown


def test_process_changes_tags_each_document_with_its_access_tier(tmp_path):
    (tmp_path / "tier-2").mkdir()
    (tmp_path / "tier-2" / "a.txt").write_text("Arsenal drew 1-1.")
    changes = FolderChanges(created=["tier-2/a.txt"], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert failures == []
    assert documents[0].access_tier == "tier-2"


def test_process_changes_isolates_an_untagged_file_without_losing_valid_ones(tmp_path):
    (tmp_path / "tier-1").mkdir()
    (tmp_path / "tier-1" / "good.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / "bad.txt").write_text("no tier folder")
    changes = FolderChanges(
        created=["tier-1/good.txt", "bad.txt"], modified=[], deleted=[]
    )

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert [doc.relative_path for doc in documents] == ["tier-1/good.txt"]
    assert [f.relative_path for f in failures] == ["bad.txt"]
    assert "no tier subfolder" in failures[0].reason


def test_process_changes_isolates_an_unknown_tier_without_losing_valid_ones(tmp_path):
    (tmp_path / "tier-1").mkdir()
    (tmp_path / "tier-1" / "good.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / "not-a-tier").mkdir()
    (tmp_path / "not-a-tier" / "bad.txt").write_text("wrong folder")
    changes = FolderChanges(
        created=["tier-1/good.txt", "not-a-tier/bad.txt"], modified=[], deleted=[]
    )

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert [doc.relative_path for doc in documents] == ["tier-1/good.txt"]
    assert [f.relative_path for f in failures] == ["not-a-tier/bad.txt"]
    assert "unknown access tier" in failures[0].reason


def test_process_changes_chunks_each_document(tmp_path):
    (tmp_path / "tier-1").mkdir()
    (tmp_path / "tier-1" / "a.txt").write_text("Arsenal drew 1-1.")
    changes = FolderChanges(created=["tier-1/a.txt"], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert failures == []
    assert len(documents[0].chunks) == 1
    assert documents[0].chunks[0].text == "Arsenal drew 1-1."
    assert documents[0].chunks[0].index == 0


def test_process_changes_respects_chunk_size_chars(tmp_path):
    block_a = "A" * 30
    block_b = "B" * 30
    (tmp_path / "tier-1").mkdir()
    (tmp_path / "tier-1" / "a.txt").write_text(f"{block_a}\n\n{block_b}")
    changes = FolderChanges(created=["tier-1/a.txt"], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=40, known_tiers=KNOWN_TIERS
    )

    assert failures == []
    assert [c.text for c in documents[0].chunks] == [block_a, block_b]


def test_process_changes_ignores_deleted_files(tmp_path):
    changes = FolderChanges(created=[], modified=[], deleted=["tier-1/c.txt"])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert documents == []
    assert failures == []


def test_process_changes_returns_empty_lists_for_no_changes(tmp_path):
    changes = FolderChanges(created=[], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert documents == []
    assert failures == []
