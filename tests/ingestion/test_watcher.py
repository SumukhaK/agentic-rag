import os

from agentic_rag.ingestion.watcher import diff_snapshots, snapshot


def test_snapshot_lists_files_with_relative_paths_and_state(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("world")

    result = snapshot(tmp_path)

    assert set(result.keys()) == {"a.txt", os.path.join("sub", "b.txt")}
    assert result["a.txt"].size == len("hello")


def test_diff_detects_created_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    before = snapshot(tmp_path)

    (tmp_path / "b.txt").write_text("new")
    after = snapshot(tmp_path)

    changes = diff_snapshots(before, after)

    assert changes.created == ["b.txt"]
    assert changes.modified == []
    assert changes.deleted == []


def test_diff_detects_deleted_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("new")
    before = snapshot(tmp_path)

    (tmp_path / "b.txt").unlink()
    after = snapshot(tmp_path)

    changes = diff_snapshots(before, after)

    assert changes.deleted == ["b.txt"]
    assert changes.created == []
    assert changes.modified == []


def test_diff_detects_modified_file(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello")
    before = snapshot(tmp_path)

    file_path.write_text("hello world")
    later_mtime = os.stat(file_path).st_mtime_ns + 1_000_000_000
    os.utime(file_path, ns=(later_mtime, later_mtime))
    after = snapshot(tmp_path)

    changes = diff_snapshots(before, after)

    assert changes.modified == ["a.txt"]
    assert changes.created == []
    assert changes.deleted == []


def test_diff_ignores_unchanged_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    before = snapshot(tmp_path)
    after = snapshot(tmp_path)

    changes = diff_snapshots(before, after)

    assert changes == diff_snapshots(before, before)
    assert changes.created == []
    assert changes.modified == []
    assert changes.deleted == []
