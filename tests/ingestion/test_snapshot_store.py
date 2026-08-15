from agentic_rag.ingestion.snapshot_store import load_snapshot, save_snapshot
from agentic_rag.ingestion.watcher import FileState


def test_load_snapshot_returns_empty_when_the_file_does_not_exist_yet(tmp_path):
    # First-ever run: nothing has been persisted yet, matching
    # run_sync_loop()'s own documented cold-start behavior of treating
    # every file in the watched folder as new.
    result = load_snapshot(tmp_path / "snapshot.json")

    assert result == {}


def test_save_then_load_round_trips_a_snapshot(tmp_path):
    snapshot = {
        "tier-1/a.txt": FileState(size=100, mtime_ns=123456789),
        "tier-2/report.md": FileState(size=42, mtime_ns=987654321),
    }
    path = tmp_path / "snapshot.json"

    save_snapshot(path, snapshot)
    result = load_snapshot(path)

    assert result == snapshot


def test_save_snapshot_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "snapshot.json"

    save_snapshot(path, {"a.txt": FileState(size=1, mtime_ns=1)})

    assert path.exists()
    assert load_snapshot(path) == {"a.txt": FileState(size=1, mtime_ns=1)}


def test_save_snapshot_overwrites_a_previous_snapshot(tmp_path):
    path = tmp_path / "snapshot.json"
    save_snapshot(path, {"old.txt": FileState(size=1, mtime_ns=1)})

    save_snapshot(path, {"new.txt": FileState(size=2, mtime_ns=2)})

    assert load_snapshot(path) == {"new.txt": FileState(size=2, mtime_ns=2)}


def test_save_snapshot_writes_atomically_leaving_no_temp_file_behind(tmp_path):
    # A crash mid-write must not leave a corrupt/truncated snapshot that
    # would be silently misread as valid (or worse, as "nothing has ever
    # been indexed") on the next startup.
    path = tmp_path / "snapshot.json"

    save_snapshot(path, {"a.txt": FileState(size=1, mtime_ns=1)})

    assert list(tmp_path.iterdir()) == [path]
