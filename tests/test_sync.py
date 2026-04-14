"""Tests for push/pull sync between local and shared MemoryStore."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.schemas.reasoning import (
    MemoryScope,
    NodeType,
    ReasoningDAG,
    ReasoningNode,
)


@pytest.fixture
def fake_embed():
    def _embed(self, texts):
        return [[float(i % 5) / 10.0 + 0.01 * j for j in range(8)] for i in range(len(texts))]
    with patch("src.store.vector_store.MemoryStore.embed", _embed):
        yield


@pytest.fixture
def store(tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    return MemoryStore(
        local_path=str(tmp_path / "local"),
        shared_path=str(tmp_path / "shared"),
    )


@pytest.fixture
def syncer(store, tmp_path):
    from src.sync.sync import Syncer, SyncLog
    log = SyncLog(db_path=str(tmp_path / "sync.db"))
    return Syncer(store=store, log=log, developer="alice")


def _dag(session_id: str, n_nodes: int = 3, scope: MemoryScope = MemoryScope.PROJECT) -> ReasoningDAG:
    return ReasoningDAG(
        session_id=session_id,
        nodes=[
            ReasoningNode(
                node_id=f"n{i}",
                node_type=NodeType.HYPOTHESIS,
                summary=f"hypothesis {session_id} {i}",
                evidence="...",
                message_range=(i, i + 1),
                confidence=0.7,
                scope=scope,
            )
            for i in range(n_nodes)
        ],
        edges=[],
    )


# ── Syncer init ──────────────────────────────────────────────────────────


def test_syncer_requires_shared_store(tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    from src.sync.sync import Syncer
    local_only = MemoryStore(local_path=str(tmp_path / "local"))
    with pytest.raises(RuntimeError):
        Syncer(store=local_only)


# ── Push ─────────────────────────────────────────────────────────────────


def test_push_moves_unpushed_to_staging(store, syncer):
    store.store_dag(_dag("s1", n_nodes=3), "proj1")

    result = syncer.push("proj1")
    assert result.pushed == 3
    assert result.errors == []

    # Local nodes are now marked pushed
    assert len(store.get_unpushed_nodes("proj1")) == 0
    # Shared staging contains them
    assert len(store.list_pending_in_staging("proj1")) == 3


def test_push_does_not_repush(store, syncer):
    store.store_dag(_dag("s1", n_nodes=3), "proj1")
    syncer.push("proj1")

    result2 = syncer.push("proj1")
    assert result2.pushed == 0
    assert len(store.list_pending_in_staging("proj1")) == 3  # unchanged


def test_push_dry_run_does_not_modify(store, syncer):
    store.store_dag(_dag("s1", n_nodes=3), "proj1")

    result = syncer.push("proj1", dry_run=True)
    assert result.pushed == 3

    # Nothing actually moved
    assert len(store.get_unpushed_nodes("proj1")) == 3
    assert len(store.list_pending_in_staging("proj1")) == 0


def test_push_records_in_log(store, syncer):
    store.store_dag(_dag("s1", n_nodes=2), "proj1")
    syncer.push("proj1")
    last = syncer.log.last_event("proj1", "push")
    assert last is not None
    assert last["count"] == 2
    assert last["actor"] == "alice"


def test_push_attaches_developer_to_staged_nodes(store, syncer):
    store.store_dag(_dag("s1", n_nodes=2), "proj1")
    syncer.push("proj1")
    pending = store.list_pending_in_staging("proj1")
    assert all(p["source_developer"] == "alice" for p in pending)


# ── Pull ─────────────────────────────────────────────────────────────────


def _seed_approved_in_shared(store, project_id, n_nodes=3, scope=MemoryScope.PROJECT):
    """Helper: store, push, then promote to shared main."""
    store.store_dag(_dag("seed", n_nodes=n_nodes, scope=scope), project_id)
    unpushed = store.get_unpushed_nodes(project_id)
    store.stage_to_shared(unpushed, developer="alice")
    pending_ids = [p["id"] for p in store.list_pending_in_staging(project_id)]
    store.promote_from_staging(pending_ids, approver="bob")


def test_pull_only_returns_approved(store, syncer):
    # Stage but DON'T approve
    store.store_dag(_dag("s1", n_nodes=3), "proj1")
    unpushed = store.get_unpushed_nodes("proj1")
    store.stage_to_shared(unpushed, developer="alice")

    # Wipe the local copies so pull has something to do
    local_ids = store.nodes_col_local.get(include=[])["ids"]
    store.nodes_col_local.delete(ids=local_ids)

    result = syncer.pull("proj1")
    # Nothing approved → nothing pulled
    assert result.pulled_project == 0
    assert result.pulled_team == 0


def test_pull_fetches_approved_nodes(store, syncer, tmp_path, fake_embed):
    # Use a fresh local store so we can pull cleanly
    from src.store.vector_store import MemoryStore
    from src.sync.sync import Syncer

    # First store, populates shared
    store.store_dag(_dag("s1", n_nodes=3), "proj1")
    unpushed = store.get_unpushed_nodes("proj1")
    store.stage_to_shared(unpushed, developer="alice")
    pending_ids = [p["id"] for p in store.list_pending_in_staging("proj1")]
    store.promote_from_staging(pending_ids, approver="bob")

    # Second developer: fresh local store, same shared
    bob_store = MemoryStore(
        local_path=str(tmp_path / "bob_local"),
        shared_path=str(tmp_path / "shared"),
    )
    bob_syncer = Syncer(store=bob_store, developer="bob")

    result = bob_syncer.pull("proj1")
    assert result.pulled_project == 3
    assert result.pulled_team == 0
    assert bob_store.node_count("proj1") == 3


def test_pull_includes_team_scope(store, syncer, tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    from src.sync.sync import Syncer

    # Seed a TEAM node under proj_other
    store.store_dag(_dag("s1", n_nodes=2, scope=MemoryScope.TEAM), "proj_other")
    unpushed = store.get_unpushed_nodes("proj_other")
    store.stage_to_shared(unpushed, developer="alice")
    pending_ids = [p["id"] for p in store.list_pending_in_staging("proj_other")]
    store.promote_from_staging(pending_ids, approver="bob")

    # Bob pulls for proj1 (different project) — should still get team nodes
    bob_store = MemoryStore(
        local_path=str(tmp_path / "bob_local"),
        shared_path=str(tmp_path / "shared"),
    )
    bob_syncer = Syncer(store=bob_store, developer="bob")
    result = bob_syncer.pull("proj1", include_team=True)
    assert result.pulled_team == 2


def test_pull_skips_team_when_disabled(store, syncer, tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    from src.sync.sync import Syncer

    store.store_dag(_dag("s1", n_nodes=2, scope=MemoryScope.TEAM), "proj_other")
    unpushed = store.get_unpushed_nodes("proj_other")
    store.stage_to_shared(unpushed, developer="alice")
    pending_ids = [p["id"] for p in store.list_pending_in_staging("proj_other")]
    store.promote_from_staging(pending_ids, approver="bob")

    bob_store = MemoryStore(
        local_path=str(tmp_path / "bob_local"),
        shared_path=str(tmp_path / "shared"),
    )
    bob_syncer = Syncer(store=bob_store, developer="bob")
    result = bob_syncer.pull("proj1", include_team=False)
    assert result.pulled_team == 0
    assert result.pulled_project == 0


def test_pull_dedupes_existing_local_nodes(store, syncer):
    # First, push and approve, leaving the original local copies in place
    store.store_dag(_dag("s1", n_nodes=3), "proj1")
    syncer.push("proj1")
    pending_ids = [p["id"] for p in store.list_pending_in_staging("proj1")]
    store.promote_from_staging(pending_ids, approver="bob")

    # Pull should find them already present locally
    result = syncer.pull("proj1")
    assert result.pulled_project == 0
    assert result.deduped == 3


# ── Status ───────────────────────────────────────────────────────────────


def test_status_reports_counts(store, syncer):
    store.store_dag(_dag("s1", n_nodes=4), "proj1")
    syncer.push("proj1")
    pending_ids = [p["id"] for p in store.list_pending_in_staging("proj1")]
    store.promote_from_staging(pending_ids[:2], approver="bob")

    s = syncer.status("proj1")
    assert s["project_id"] == "proj1"
    assert s["local_nodes"] == 4
    assert s["unpushed_nodes"] == 0
    assert s["shared_approved"] == 2
    assert s["pending_review"] == 2
    assert s["last_push"] is not None


# ── Sync log ─────────────────────────────────────────────────────────────


def test_sync_log_records_events(tmp_path):
    from src.sync.sync import SyncLog
    log = SyncLog(db_path=str(tmp_path / "sync.db"))

    log.record("proj1", "push", count=5, actor="alice", detail="ok")
    log.record("proj1", "pull", count=3, actor="alice", detail="ok")
    log.record("proj1", "push", count=2, actor="alice", detail="ok")

    history = log.history("proj1")
    assert len(history) == 3

    last_push = log.last_event("proj1", "push")
    assert last_push["count"] == 2  # most recent push


def test_sync_log_fail_safe(tmp_path):
    """A bad path doesn't crash sync operations."""
    from src.sync.sync import SyncLog
    log = SyncLog(db_path="/nonexistent/path/that/does/not/exist/sync.db")
    # Should not raise
    log.record("proj1", "push", count=1)
    assert log.history("proj1") == []
