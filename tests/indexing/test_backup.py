from pathlib import Path

import pytest

from agentic_rag.indexing.backup import backup_qdrant_storage, list_backups


def _write(path: Path, content: str = "data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_backup_qdrant_storage_copies_every_file(tmp_path):
    storage = tmp_path / "qdrant"
    _write(storage / "collection" / "storage.sqlite", "vectors")
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"

    result = backup_qdrant_storage(storage, backup_dir, retention_count=3)

    assert (result / "collection" / "storage.sqlite").read_text() == "vectors"
    assert (result / "meta.json").read_text() == "{}"


def test_backup_qdrant_storage_raises_when_storage_path_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        backup_qdrant_storage(tmp_path / "nonexistent", tmp_path / "backups", retention_count=3)


def test_backup_qdrant_storage_creates_a_new_timestamped_directory_each_call(tmp_path):
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"

    first = backup_qdrant_storage(storage, backup_dir, retention_count=3)
    second = backup_qdrant_storage(storage, backup_dir, retention_count=3)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_backup_qdrant_storage_prunes_backups_beyond_retention_count(tmp_path):
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"

    for _ in range(5):
        backup_qdrant_storage(storage, backup_dir, retention_count=2)

    assert len(list_backups(backup_dir)) == 2


def test_backup_qdrant_storage_keeps_the_most_recent_backups_when_pruning(tmp_path):
    storage = tmp_path / "qdrant"
    backup_dir = tmp_path / "backups"

    _write(storage / "meta.json", "v1")
    first = backup_qdrant_storage(storage, backup_dir, retention_count=2)
    _write(storage / "meta.json", "v2")
    second = backup_qdrant_storage(storage, backup_dir, retention_count=2)
    _write(storage / "meta.json", "v3")
    third = backup_qdrant_storage(storage, backup_dir, retention_count=2)

    remaining = list_backups(backup_dir)
    assert first not in remaining
    assert second in remaining
    assert third in remaining


def test_backup_qdrant_storage_does_not_prune_when_retention_count_is_not_exceeded(tmp_path):
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"

    backup_qdrant_storage(storage, backup_dir, retention_count=5)
    backup_qdrant_storage(storage, backup_dir, retention_count=5)

    assert len(list_backups(backup_dir)) == 2


def test_list_backups_returns_empty_list_when_backup_dir_does_not_exist(tmp_path):
    assert list_backups(tmp_path / "never_backed_up") == []


def test_list_backups_ignores_leftover_tmp_directories_from_an_interrupted_copy(tmp_path):
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"
    backup_qdrant_storage(storage, backup_dir, retention_count=3)
    # Simulate a crash mid-copy on a later run: a .tmp directory left behind.
    (backup_dir / "20260101T000000000000Z.tmp").mkdir()

    backups = list_backups(backup_dir)

    assert all(not p.name.endswith(".tmp") for p in backups)


def test_backup_qdrant_storage_overwrites_a_leftover_tmp_directory_from_a_prior_crash(tmp_path):
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True)
    stale_tmp = backup_dir / "stale.tmp"
    stale_tmp.mkdir()
    (stale_tmp / "partial_file").write_text("incomplete")

    # Not asserting anything about the specific stale name - just that a
    # fresh backup call still succeeds cleanly regardless of leftover state.
    result = backup_qdrant_storage(storage, backup_dir, retention_count=3)

    assert result.exists()
