"""Session watcher tests — Phase 4.1 exit criteria."""
import json
import time
import pytest
from pathlib import Path

from src.ingestion.watcher import SessionWatcher, WatchEvent


@pytest.fixture
def watch_dir(tmp_path):
    """Create a mock Claude Code projects directory structure."""
    project_dir = tmp_path / "-Users-test-Downloads-myproject"
    project_dir.mkdir()
    return tmp_path


@pytest.fixture
def store_dir(tmp_path):
    return tmp_path / "store"


def _write_session(project_dir: Path, name: str = "session1.jsonl", content: str = None):
    """Write a minimal JSONL session file."""
    if content is None:
        content = (
            '{"type":"user","message":{"role":"user","content":"hello"}}\n'
            '{"type":"assistant","message":{"role":"assistant","content":"Hi! I can help you."}}\n'
        )
    path = project_dir / name
    path.write_text(content)
    # Set mtime to past so min_file_age is satisfied
    old_time = time.time() - 120
    import os
    os.utime(path, (old_time, old_time))
    return path


def test_scan_detects_new_session(watch_dir):
    """Watcher detects a new JSONL session file."""
    project_dir = watch_dir / "-Users-test-Downloads-myproject"
    _write_session(project_dir)

    watcher = SessionWatcher(watch_dir=watch_dir, min_file_age=0)
    events = watcher.scan()

    assert len(events) == 1
    assert events[0].event_type == "new"
    assert events[0].path.name == "session1.jsonl"
    assert events[0].project_dir_name == "-Users-test-Downloads-myproject"


def test_scan_ignores_empty_files(watch_dir):
    """Watcher ignores empty JSONL files."""
    project_dir = watch_dir / "-Users-test-Downloads-myproject"
    _write_session(project_dir, content="")

    watcher = SessionWatcher(watch_dir=watch_dir, min_file_age=0)
    events = watcher.scan()

    assert len(events) == 0


def test_scan_ignores_recently_modified(watch_dir):
    """Watcher ignores files modified less than min_file_age seconds ago."""
    project_dir = watch_dir / "-Users-test-Downloads-myproject"
    path = project_dir / "session1.jsonl"
    path.write_text('{"type":"user","message":{"role":"user","content":"hello"}}\n')
    # Don't set mtime to past — leave it as current

    watcher = SessionWatcher(watch_dir=watch_dir, min_file_age=60)
    events = watcher.scan()

    assert len(events) == 0


def test_scan_no_duplicate_on_second_scan(watch_dir):
    """Second scan doesn't re-report files already tracked."""
    project_dir = watch_dir / "-Users-test-Downloads-myproject"
    _write_session(project_dir)

    watcher = SessionWatcher(watch_dir=watch_dir, min_file_age=0)

    events1 = watcher.scan()
    assert len(events1) == 1

    # Mark as ingested
    watcher.mark_ingested(events1[0].path)

    events2 = watcher.scan()
    assert len(events2) == 0


def test_scan_detects_modified_after_ingest(watch_dir):
    """Watcher detects when a previously ingested file is modified."""
    project_dir = watch_dir / "-Users-test-Downloads-myproject"
    path = _write_session(project_dir)

    watcher = SessionWatcher(watch_dir=watch_dir, min_file_age=0)
    events = watcher.scan()
    watcher.mark_ingested(events[0].path)

    # Modify the file after ingest — mtime must be after ingest_time
    import os
    path.write_text(path.read_text() + '{"type":"user","message":{"role":"user","content":"more"}}\n')
    # Set mtime well into the future so it's clearly after ingest_time
    future_time = time.time() + 100
    os.utime(path, (future_time, future_time))

    events2 = watcher.scan()
    assert len(events2) == 1
    assert events2[0].event_type == "modified"


def test_scan_multiple_projects(watch_dir):
    """Watcher scans across multiple project directories."""
    proj1 = watch_dir / "-Users-test-project-alpha"
    proj1.mkdir()
    proj2 = watch_dir / "-Users-test-project-beta"
    proj2.mkdir()

    _write_session(proj1, "s1.jsonl")
    _write_session(proj2, "s2.jsonl")

    watcher = SessionWatcher(watch_dir=watch_dir, min_file_age=0)
    events = watcher.scan()

    assert len(events) == 2
    dirs = {e.project_dir_name for e in events}
    assert "-Users-test-project-alpha" in dirs
    assert "-Users-test-project-beta" in dirs


def test_derive_project_id():
    """Project ID derivation from Claude Code directory names."""
    watcher = SessionWatcher()

    assert watcher.derive_project_id("-Users-sazankhalid-Downloads-cmm") == "cmm"
    assert watcher.derive_project_id("-Users-test-Downloads-mcp-gateway") == "gateway"

    # With explicit map
    watcher.project_map["-Users-test-myapp"] = "my-app"
    assert watcher.derive_project_id("-Users-test-myapp") == "my-app"


def test_derive_project_id_with_map():
    """Project map takes precedence over derivation."""
    watcher = SessionWatcher(project_map={
        "-Users-sazankhalid-Downloads-cmm": "supply-chain"
    })
    assert watcher.derive_project_id("-Users-sazankhalid-Downloads-cmm") == "supply-chain"


def test_state_persistence(watch_dir, tmp_path):
    """Watcher state persists across restarts."""
    store = tmp_path / "store"
    store.mkdir()

    project_dir = watch_dir / "-Users-test-Downloads-myproject"
    path = _write_session(project_dir)

    # First watcher instance
    w1 = SessionWatcher(watch_dir=watch_dir, store_path=str(store), min_file_age=0)
    events = w1.scan()
    assert len(events) == 1
    w1.mark_ingested(events[0].path)

    # New watcher instance — should load state
    w2 = SessionWatcher(watch_dir=watch_dir, store_path=str(store), min_file_age=0)
    w2._load_state()
    events2 = w2.scan()
    assert len(events2) == 0  # Already ingested


def test_nonexistent_watch_dir(tmp_path):
    """Watcher handles nonexistent watch directory gracefully."""
    watcher = SessionWatcher(watch_dir=tmp_path / "nonexistent", min_file_age=0)
    events = watcher.scan()
    assert events == []
