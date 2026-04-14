"""Tests for dual-store MemoryStore (local + shared)."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.schemas.reasoning import (
    MemoryScope,
    NodeType,
    ReasoningDAG,
    ReasoningEdge,
    ReasoningNode,
)


@pytest.fixture
def fake_embed():
    """Patch OpenAI embedding to return deterministic vectors."""
    def _embed(self, texts):
        return [[float(i % 7) / 10.0] * 8 for i in range(len(texts))]
    with patch("src.store.vector_store.MemoryStore.embed", _embed):
        yield


@pytest.fixture
def local_store(tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    return MemoryStore(local_path=str(tmp_path / "local"))


@pytest.fixture
def shared_store(tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    return MemoryStore(
        local_path=str(tmp_path / "local"),
        shared_path=str(tmp_path / "shared"),
    )


def _dag(session_id: str, scope: MemoryScope = MemoryScope.PROJECT) -> ReasoningDAG:
    return ReasoningDAG(
        session_id=session_id,
        nodes=[
            ReasoningNode(
                node_id=f"n{i}",
                node_type=NodeType.HYPOTHESIS,
                summary=f"hypothesis {i}",
                evidence="...",
                message_range=(i, i + 1),
                confidence=0.7,
                scope=scope,
            )
            for i in range(3)
        ],
        edges=[],
        pivot_nodes=[],
    )


# ── Backwards compatibility ──────────────────────────────────────────────


def test_legacy_persist_dir_kwarg_still_works(tmp_path, fake_embed):
    """The old persist_dir= kwarg must still work."""
    from src.store.vector_store import MemoryStore
    store = MemoryStore(persist_dir=str(tmp_path / "legacy"))
    assert store._local_path.endswith("legacy")
    assert not store.has_shared
    # Aliases work
    assert store.client is store.client_local
    assert store.nodes_col is store.nodes_col_local


def test_local_only_mode_default(local_store):
    """No shared_path → has_shared is False."""
    assert local_store.has_shared is False
    assert local_store.client_shared is None
    assert local_store.nodes_col_shared is None


def test_missing_local_path_raises(tmp_path, fake_embed):
    from src.store.vector_store import MemoryStore
    with pytest.raises(ValueError):
        MemoryStore()


# ── Shared mode setup ────────────────────────────────────────────────────


def test_shared_mode_creates_both_clients(shared_store):
    assert shared_store.has_shared
    assert shared_store.client_local is not None
    assert shared_store.client_shared is not None
    assert shared_store.staging_col_shared is not None


def test_get_collection_dispatch(shared_store):
    assert shared_store.get_collection("local") is shared_store.nodes_col_local
    assert shared_store.get_collection("shared") is shared_store.nodes_col_shared
    assert shared_store.get_collection("staging") is shared_store.staging_col_shared

    with pytest.raises(ValueError):
        shared_store.get_collection("nonsense")


def test_get_collection_shared_fails_in_local_mode(local_store):
    with pytest.raises(RuntimeError):
        local_store.get_collection("shared")
    with pytest.raises(RuntimeError):
        local_store.get_collection("staging")


# ── Storage writes scope and sync metadata ──────────────────────────────


def test_store_dag_writes_scope_metadata(local_store):
    dag = _dag("s1", scope=MemoryScope.TEAM)
    local_store.store_dag(dag, "proj1")

    results = local_store.nodes_col_local.get(
        where={"project_id": "proj1"},
        include=["metadatas"],
    )
    assert len(results["metadatas"]) == 3
    for meta in results["metadatas"]:
        assert meta["scope"] == "team"
        assert meta["pushed_at"] == ""
        assert meta["approved"] is False


# ── Unpushed node tracking ──────────────────────────────────────────────


def test_get_unpushed_returns_only_unpushed(local_store):
    dag = _dag("s1")
    local_store.store_dag(dag, "proj1")

    unpushed = local_store.get_unpushed_nodes("proj1")
    assert len(unpushed) == 3

    # Mark them pushed
    ids = [n["id"] for n in unpushed]
    local_store.mark_pushed(ids, when="2026-01-01T00:00:00")

    unpushed_after = local_store.get_unpushed_nodes("proj1")
    assert len(unpushed_after) == 0


def test_mark_pushed_idempotent(local_store):
    dag = _dag("s1")
    local_store.store_dag(dag, "proj1")
    ids = [n["id"] for n in local_store.get_unpushed_nodes("proj1")]

    local_store.mark_pushed(ids, when="2026-01-01T00:00:00")
    local_store.mark_pushed(ids, when="2026-01-02T00:00:00")  # second call

    results = local_store.nodes_col_local.get(ids=ids, include=["metadatas"])
    for meta in results["metadatas"]:
        assert meta["pushed_at"] == "2026-01-02T00:00:00"


# ── Staging operations ──────────────────────────────────────────────────


def test_stage_to_shared_and_list_pending(shared_store):
    dag = _dag("s1")
    shared_store.store_dag(dag, "proj1")
    unpushed = shared_store.get_unpushed_nodes("proj1")

    count = shared_store.stage_to_shared(unpushed, developer="alice")
    assert count == 3

    pending = shared_store.list_pending_in_staging("proj1")
    assert len(pending) == 3
    assert all(p["source_developer"] == "alice" for p in pending)
    assert all(p["approved"] is False for p in pending)


def test_promote_from_staging_moves_to_main(shared_store):
    dag = _dag("s1")
    shared_store.store_dag(dag, "proj1")
    unpushed = shared_store.get_unpushed_nodes("proj1")
    shared_store.stage_to_shared(unpushed, developer="alice")

    pending = shared_store.list_pending_in_staging("proj1")
    pending_ids = [p["id"] for p in pending]

    promoted = shared_store.promote_from_staging(pending_ids, approver="bob")
    assert promoted == 3

    # Staging is now empty
    assert shared_store.list_pending_in_staging("proj1") == []

    # Main contains the approved ones
    approved = shared_store.list_approved_shared(project_id="proj1")
    assert len(approved) == 3
    for n in approved:
        assert n["approved"] is True
        assert n["approved_by"] == "bob"


def test_promote_with_scope_override(shared_store):
    dag = _dag("s1", scope=MemoryScope.PROJECT)
    shared_store.store_dag(dag, "proj1")
    unpushed = shared_store.get_unpushed_nodes("proj1")
    shared_store.stage_to_shared(unpushed, developer="alice")

    pending_ids = [p["id"] for p in shared_store.list_pending_in_staging("proj1")]
    overrides = {pending_ids[0]: "team"}
    shared_store.promote_from_staging(pending_ids, approver="bob", scope_overrides=overrides)

    approved = shared_store.list_approved_shared(project_id="proj1", include_team=True)
    scopes = sorted(n["scope"] for n in approved)
    assert "team" in scopes


def test_reject_in_staging(shared_store):
    dag = _dag("s1")
    shared_store.store_dag(dag, "proj1")
    unpushed = shared_store.get_unpushed_nodes("proj1")
    shared_store.stage_to_shared(unpushed, developer="alice")

    ids = [p["id"] for p in shared_store.list_pending_in_staging("proj1")]
    rejected = shared_store.reject_in_staging(ids[:1], reviewer="bob", reason="hallucinated")
    assert rejected == 1

    # The rejected one is not in pending anymore (approved=False but rejected=True)
    pending = shared_store.list_pending_in_staging("proj1")
    # list_pending_in_staging returns all unapproved — including rejected
    rejected_entry = [p for p in pending if p["id"] == ids[0]]
    assert rejected_entry  # still present for audit
    assert rejected_entry[0]["rejected"] is True


# ── Pull operation ──────────────────────────────────────────────────────


def test_list_approved_includes_team_scope(shared_store):
    project_dag = _dag("s1", scope=MemoryScope.PROJECT)
    shared_store.store_dag(project_dag, "proj1")

    team_dag = ReasoningDAG(
        session_id="s2",
        nodes=[ReasoningNode(
            node_id="t0", node_type=NodeType.DISCOVERY,
            summary="general team knowledge", evidence="...",
            message_range=(0, 1), scope=MemoryScope.TEAM,
        )],
        edges=[],
    )
    shared_store.store_dag(team_dag, "proj2")

    # Stage and approve everything
    unpushed = shared_store.get_unpushed_nodes("proj1") + shared_store.get_unpushed_nodes("proj2")
    shared_store.stage_to_shared(unpushed, developer="alice")
    pending_ids = [p["id"] for p in shared_store.list_pending_in_staging()]
    shared_store.promote_from_staging(pending_ids, approver="bob")

    # Pull for proj1 — should include team-scope from proj2
    approved = shared_store.list_approved_shared(project_id="proj1", include_team=True)
    project_ids = sorted(set(n["project_id"] for n in approved))
    assert "proj1" in project_ids
    # The team node from proj2 is included
    scopes = [n["scope"] for n in approved]
    assert "team" in scopes


def test_upsert_pulled_nodes_marks_pushed(local_store):
    """Pulled nodes get pushed_at set so they don't bounce back."""
    pulled = [{
        "id": "remote::s1::n0",
        "summary": "team insight",
        "embedding": [0.1] * 8,
        "project_id": "other",
        "session_id": "s1",
        "node_id": "n0",
        "node_type": "discovery",
        "confidence": 0.9,
        "msg_start": 0,
        "msg_end": 1,
        "is_pivot": False,
        "scope": "team",
        "approved": True,
        "approved_by": "bob",
        "source_developer": "alice",
    }]
    local_store.upsert_pulled_nodes(pulled, project_id="proj1")

    unpushed_after = local_store.get_unpushed_nodes("other")
    # Team-scope node retains its original project_id
    assert len(unpushed_after) == 0  # already marked pushed
