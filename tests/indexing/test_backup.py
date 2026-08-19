from pathlib import Path

import pytest
from qdrant_client.models import PointStruct

from agentic_rag.indexing.backup import backup_qdrant_storage, list_backups
from agentic_rag.indexing.qdrant_setup import ensure_collection, get_client


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


def test_backup_qdrant_storage_succeeds_despite_an_unrelated_leftover_tmp_directory(tmp_path):
    # An unrelated .tmp left behind by a prior crash (a different name -
    # not the one THIS call will generate) must not make a fresh backup
    # call fail. This does not claim the leftover itself gets swept up -
    # see test_backup_qdrant_storage_removes_its_own_tmp_directory_on_failure
    # for the actual cleanup guarantee this function provides.
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True)
    stale_tmp = backup_dir / "stale.tmp"
    stale_tmp.mkdir()
    (stale_tmp / "partial_file").write_text("incomplete")

    result = backup_qdrant_storage(storage, backup_dir, retention_count=3)

    assert result.exists()


def test_backup_qdrant_storage_removes_its_own_tmp_directory_on_failure(tmp_path, monkeypatch):
    # A failed copy (disk full, a permissions error, anything) must not
    # leave a large, silently-orphaned partial copy on disk forever -
    # list_backups()/_prune_old_backups() never touch .tmp directories by
    # design (an incomplete backup must never be counted as restorable),
    # so cleanup on failure has to happen here, immediately.
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("shutil.copytree", _boom)

    with pytest.raises(OSError):
        backup_qdrant_storage(storage, backup_dir, retention_count=3)

    leftover_tmp_dirs = list(backup_dir.glob("*.tmp")) if backup_dir.exists() else []
    assert leftover_tmp_dirs == []


def test_backup_qdrant_storage_excludes_the_lock_file_from_the_copy(tmp_path):
    # Qdrant's local mode holds an exclusive OS-level lock on `.lock` for
    # the whole life of a QdrantClient - shutil.copytree() trying to open
    # it too fails on Windows (reproduced directly against a real, still-
    # open client during review). The fix is to never attempt to copy it
    # in the first place; this test simulates the file being present
    # (without needing a real live-locked client, which the qdrant_setup
    # tests already cover) and confirms it's deliberately skipped.
    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    _write(storage / ".lock", "lock-marker")
    backup_dir = tmp_path / "backups"

    result = backup_qdrant_storage(storage, backup_dir, retention_count=3)

    assert (result / "meta.json").exists()
    assert not (result / ".lock").exists()


def test_prune_old_backups_removes_every_backup_when_retention_count_is_zero(tmp_path):
    # backups[:-0] means backups[:0] (an empty slice) in Python, not "the
    # whole list" - a retention_count of 0 must still prune everything,
    # not silently keep every backup forever. Settings itself rejects
    # retention_count <= 0 (Field(gt=0)), so this exercises the pruning
    # helper directly rather than going through Settings validation.
    from agentic_rag.indexing.backup import _prune_old_backups

    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"
    for _ in range(3):
        backup_qdrant_storage(storage, backup_dir, retention_count=10)
    assert len(list_backups(backup_dir)) == 3

    _prune_old_backups(backup_dir, 0)

    assert list_backups(backup_dir) == []


def test_backup_qdrant_storage_gives_the_final_directory_a_unique_name_even_on_a_forced_timestamp_collision(
    tmp_path, monkeypatch
):
    # A same-microsecond timestamp collision must not make the second
    # call's Path.replace() collide with the first call's already-existing
    # final directory - both the tmp AND final names need the same unique
    # suffix, not just the tmp name, or the second call raises instead of
    # producing a second, genuinely distinct backup.
    import agentic_rag.indexing.backup as backup_module

    storage = tmp_path / "qdrant"
    _write(storage / "meta.json", "{}")
    backup_dir = tmp_path / "backups"

    frozen = "20260101T000000000000"

    class _FrozenDatetime:
        @staticmethod
        def now(tz):
            class _Stamped:
                @staticmethod
                def strftime(fmt):
                    return frozen

            return _Stamped()

    monkeypatch.setattr(backup_module, "datetime", _FrozenDatetime)

    first = backup_qdrant_storage(storage, backup_dir, retention_count=5)
    second = backup_qdrant_storage(storage, backup_dir, retention_count=5)

    assert first != second
    assert first.exists()
    assert second.exists()


# --- Real Qdrant client: the actual production scenario --------------------


def test_backup_qdrant_storage_succeeds_while_a_live_client_still_holds_the_storage_open(tmp_path):
    # The scenario that actually matters: in production, ingestion/
    # scheduler.py's run_sync_loop() calls backup_qdrant_storage() while
    # api/app.py's lifespan-held QdrantClient is still open against the
    # exact same storage path for the whole life of the process - it is
    # never closed first. A test that closes the client before backing up
    # (as an earlier version of this test suite effectively did, by never
    # exercising this at all) does not exercise the real call pattern and
    # would have missed the .lock-file collision this test regression-
    # tests directly.
    storage = tmp_path / "qdrant"
    backup_dir = tmp_path / "backups"
    client = get_client(storage)
    ensure_collection(client, collection_name="documents", vector_size=4)
    client.upsert(
        collection_name="documents",
        points=[PointStruct(id=1, vector={"dense": [0.1, 0.2, 0.3, 0.4]}, payload={"path": "a.md"})],
    )

    try:
        # client is deliberately still open here - this is the point.
        result = backup_qdrant_storage(storage, backup_dir, retention_count=3)
    finally:
        client.close()

    assert result.exists()


def test_backup_qdrant_storage_produces_a_genuinely_restorable_backup(tmp_path):
    # Codifies the actual restore procedure documented in
    # docs/REQUIREMENTS.md and indexing/backup.py's own docstring: point
    # get_client() at the backup directory instead of the live storage
    # path. A real write -> backup -> fresh-client-read roundtrip, not
    # just "the files got copied" - proves the backup is actually usable,
    # not merely present.
    storage = tmp_path / "qdrant"
    backup_dir = tmp_path / "backups"
    client = get_client(storage)
    ensure_collection(client, collection_name="documents", vector_size=4)
    client.upsert(
        collection_name="documents",
        points=[PointStruct(id=1, vector={"dense": [0.1, 0.2, 0.3, 0.4]}, payload={"path": "a.md"})],
    )
    try:
        backup_path = backup_qdrant_storage(storage, backup_dir, retention_count=3)
    finally:
        client.close()

    restored_client = get_client(backup_path)
    try:
        assert restored_client.count("documents").count == 1
        points = restored_client.retrieve("documents", ids=[1])
        assert points[0].payload == {"path": "a.md"}
    finally:
        restored_client.close()
