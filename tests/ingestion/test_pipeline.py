from agentic_rag.ingestion.pipeline import process_changes
from agentic_rag.ingestion.watcher import FolderChanges
from tests.access_tiers import ACCESS_TIERS, TIER_EMPLOYEE, TIER_MANAGER



def test_process_changes_converts_created_and_modified_files(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "a.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / TIER_MANAGER).mkdir()
    (tmp_path / TIER_MANAGER / "b.txt").write_text("Chelsea won 3-0.")
    changes = FolderChanges(
        created=["employee/a.txt"], modified=["manager/b.txt"], deleted=[]
    )

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert failures == []
    assert [doc.relative_path for doc in documents] == ["employee/a.txt", "manager/b.txt"]
    assert "Arsenal drew 1-1." in documents[0].markdown
    assert "Chelsea won 3-0." in documents[1].markdown


def test_process_changes_tags_each_document_with_its_access_tier(tmp_path):
    (tmp_path / TIER_MANAGER).mkdir()
    (tmp_path / TIER_MANAGER / "a.txt").write_text("Arsenal drew 1-1.")
    changes = FolderChanges(created=["manager/a.txt"], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert failures == []
    assert documents[0].access_tier == TIER_MANAGER


def test_process_changes_isolates_an_untagged_file_without_losing_valid_ones(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "good.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / "bad.txt").write_text("no tier folder")
    changes = FolderChanges(
        created=["employee/good.txt", "bad.txt"], modified=[], deleted=[]
    )

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert [doc.relative_path for doc in documents] == ["employee/good.txt"]
    assert [f.relative_path for f in failures] == ["bad.txt"]
    assert "no tier subfolder" in failures[0].reason


def test_process_changes_isolates_an_unknown_tier_without_losing_valid_ones(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "good.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / "not-a-tier").mkdir()
    (tmp_path / "not-a-tier" / "bad.txt").write_text("wrong folder")
    changes = FolderChanges(
        created=["employee/good.txt", "not-a-tier/bad.txt"], modified=[], deleted=[]
    )

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert [doc.relative_path for doc in documents] == ["employee/good.txt"]
    assert [f.relative_path for f in failures] == ["not-a-tier/bad.txt"]
    assert "unknown access tier" in failures[0].reason


def test_process_changes_isolates_a_conversion_failure_without_losing_valid_ones(
    tmp_path,
):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "good.txt").write_text("Arsenal drew 1-1.")
    # employee/missing.txt is validly tagged but doesn't exist on disk, simulating
    # a file that vanished (or was never actually written) between the watcher
    # snapshotting it and the pipeline trying to convert it.
    changes = FolderChanges(
        created=["employee/good.txt", "employee/missing.txt"], modified=[], deleted=[]
    )

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert [doc.relative_path for doc in documents] == ["employee/good.txt"]
    assert [f.relative_path for f in failures] == ["employee/missing.txt"]


def test_process_changes_reports_a_blank_document_as_a_validation_failure(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    # An all-whitespace file converts to markdown with no usable content, so
    # chunk_markdown produces zero chunks - that should be a reported
    # failure, not a document silently entering the index with nothing in it.
    (tmp_path / TIER_EMPLOYEE / "blank.txt").write_text("   \n\n   ")
    changes = FolderChanges(created=["employee/blank.txt"], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert documents == []
    assert [f.relative_path for f in failures] == ["employee/blank.txt"]
    assert "no chunks" in failures[0].reason


def test_process_changes_prefixes_failure_reason_with_the_exception_type(tmp_path):
    # The exception class name lets an operator (or Phase 7's scheduler)
    # tell a genuine data-quality failure (DocumentValidationError,
    # UntaggedDocumentError, ...) apart from an unexpected bug - both get
    # caught by the same per-file isolation, but they shouldn't look
    # identical in the failure reason.
    (tmp_path / "a.txt").write_text("no tier folder")
    changes = FolderChanges(created=["a.txt"], modified=[], deleted=[])

    _, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert failures[0].reason.startswith("UntaggedDocumentError: ")


def test_process_changes_chunks_each_document(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "a.txt").write_text("Arsenal drew 1-1.")
    changes = FolderChanges(created=["employee/a.txt"], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert failures == []
    assert len(documents[0].chunks) == 1
    assert documents[0].chunks[0].text == "Arsenal drew 1-1."
    assert documents[0].chunks[0].index == 0


def test_process_changes_respects_chunk_size_chars(tmp_path):
    block_a = "A" * 30
    block_b = "B" * 30
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "a.txt").write_text(f"{block_a}\n\n{block_b}")
    changes = FolderChanges(created=["employee/a.txt"], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=40, known_tiers=ACCESS_TIERS
    )

    assert failures == []
    assert [c.text for c in documents[0].chunks] == [block_a, block_b]


def test_process_changes_ignores_deleted_files(tmp_path):
    changes = FolderChanges(created=[], modified=[], deleted=["employee/c.txt"])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert documents == []
    assert failures == []


def test_process_changes_returns_empty_lists_for_no_changes(tmp_path):
    changes = FolderChanges(created=[], modified=[], deleted=[])

    documents, failures = process_changes(
        tmp_path, changes, chunk_size_chars=2000, known_tiers=ACCESS_TIERS
    )

    assert documents == []
    assert failures == []
