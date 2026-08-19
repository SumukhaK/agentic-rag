import os

from agentic_rag.ingestion.sync import sync_folder
from agentic_rag.ingestion.watcher import snapshot
from access_tiers import ACCESS_TIERS, TIER_EMPLOYEE

KNOWN_TIERS = ACCESS_TIERS
TIER_1_A = os.path.join(TIER_EMPLOYEE, "a.txt")
TIER_1_GOOD = os.path.join(TIER_EMPLOYEE, "good.txt")


def test_sync_folder_detects_and_converts_a_new_document(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "a.txt").write_text("Arsenal drew 1-1.")

    result = sync_folder(
        tmp_path, previous_snapshot={}, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert [doc.relative_path for doc in result.documents] == [TIER_1_A]
    assert result.failures == []
    assert result.deleted == []


def test_sync_folder_reports_a_deleted_document(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    file_path = tmp_path / TIER_EMPLOYEE / "a.txt"
    file_path.write_text("Arsenal drew 1-1.")
    previous = snapshot(tmp_path)

    file_path.unlink()

    result = sync_folder(
        tmp_path,
        previous_snapshot=previous,
        chunk_size_chars=2000,
        known_tiers=KNOWN_TIERS,
    )

    assert result.deleted == [TIER_1_A]
    assert result.documents == []
    assert result.failures == []


def test_sync_folder_reconverts_a_modified_document(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    file_path = tmp_path / TIER_EMPLOYEE / "a.txt"
    file_path.write_text("Arsenal drew 1-1.")
    previous = snapshot(tmp_path)

    file_path.write_text("Arsenal won 3-0.")

    result = sync_folder(
        tmp_path,
        previous_snapshot=previous,
        chunk_size_chars=2000,
        known_tiers=KNOWN_TIERS,
    )

    assert [doc.relative_path for doc in result.documents] == [TIER_1_A]
    assert "Arsenal won 3-0." in result.documents[0].markdown
    assert result.deleted == []


def test_sync_folder_returns_the_current_snapshot_for_the_next_cycle(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "a.txt").write_text("Arsenal drew 1-1.")

    result = sync_folder(
        tmp_path, previous_snapshot={}, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert result.current_snapshot == snapshot(tmp_path)


def test_sync_folder_isolates_a_tagging_failure_from_valid_documents(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "good.txt").write_text("Arsenal drew 1-1.")
    (tmp_path / "bad.txt").write_text("no tier folder")

    result = sync_folder(
        tmp_path, previous_snapshot={}, chunk_size_chars=2000, known_tiers=KNOWN_TIERS
    )

    assert [doc.relative_path for doc in result.documents] == [TIER_1_GOOD]
    assert [f.relative_path for f in result.failures] == ["bad.txt"]


def test_sync_folder_reports_nothing_when_folder_is_unchanged(tmp_path):
    (tmp_path / TIER_EMPLOYEE).mkdir()
    (tmp_path / TIER_EMPLOYEE / "a.txt").write_text("Arsenal drew 1-1.")
    previous = snapshot(tmp_path)

    result = sync_folder(
        tmp_path,
        previous_snapshot=previous,
        chunk_size_chars=2000,
        known_tiers=KNOWN_TIERS,
    )

    assert result.documents == []
    assert result.failures == []
    assert result.deleted == []
