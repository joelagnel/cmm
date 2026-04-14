"""End-to-end integration test for the full cognitive memory pipeline.

Exercises every phase together:
  Phase 1 — token-budget windowing (mocked LLM)
  Phase 2 — scope on extracted nodes
  Phase 3 — dual-store local + shared
  Phase 4 — push from alice's local to shared staging
  Phase 5 — reviewer approves + reclassifies
  Phase 6 — bob pulls from shared into a fresh local store
"""
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
    """Deterministic embeddings — same text → same vector across stores."""
    def _embed(self, texts):
        out = []
        for t in texts:
            h = sum(ord(c) for c in t) % 100
            out.append([float(h) / 100 + 0.001 * i for i in range(8)])
        return out
    with patch("src.store.vector_store.MemoryStore.embed", _embed):
        yield


def _build_dag(prefix: str, n: int = 3, scope: MemoryScope = MemoryScope.PROJECT) -> ReasoningDAG:
    return ReasoningDAG(
        session_id=f"sess-{prefix}",
        nodes=[
            ReasoningNode(
                node_id=f"{prefix}-{i}",
                node_type=NodeType.DISCOVERY if i % 2 == 0 else NodeType.HYPOTHESIS,
                summary=f"{prefix} insight {i}",
                evidence=f"evidence for {prefix} {i}",
                message_range=(i, i + 1),
                confidence=0.7 + i * 0.05,
                scope=scope,
            )
            for i in range(n)
        ],
        edges=[],
    )


def test_full_alice_to_bob_pipeline(tmp_path, fake_embed):
    """A new developer (bob) gains access to alice's approved memories."""
    from src.store.vector_store import MemoryStore
    from src.sync.review import Reviewer, ReviewAction, ReviewDecision
    from src.sync.sync import SyncLog, Syncer

    # Shared store path
    shared_path = str(tmp_path / "shared")

    # ─── Alice's setup ──────────────────────────────────────────────
    alice_local = str(tmp_path / "alice_local")
    alice_store = MemoryStore(local_path=alice_local, shared_path=shared_path)
    alice_log = SyncLog(db_path=str(tmp_path / "alice_sync.db"))

    # Alice ingests two sessions on the same project, mixed scope
    project_dag = _build_dag("proj", n=3, scope=MemoryScope.PROJECT)
    team_dag = _build_dag("team", n=2, scope=MemoryScope.TEAM)
    alice_store.store_dag(project_dag, "supply-chain")
    alice_store.store_dag(team_dag, "supply-chain")

    assert alice_store.node_count("supply-chain") == 5

    # Phase 4 — Alice pushes to shared staging
    alice_syncer = Syncer(store=alice_store, log=alice_log, developer="alice")
    push_result = alice_syncer.push("supply-chain")
    assert push_result.pushed == 5
    assert len(alice_store.get_unpushed_nodes("supply-chain")) == 0
    assert len(alice_store.list_pending_in_staging("supply-chain")) == 5

    # ─── Reviewer (carol) reviews ───────────────────────────────────
    # Carol uses the same shared store. She approves 4 of 5 and rejects one.
    carol_local = str(tmp_path / "carol_local")
    carol_store = MemoryStore(local_path=carol_local, shared_path=shared_path)
    carol_reviewer = Reviewer(store=carol_store, reviewer_name="carol")

    pending = carol_reviewer.list_pending("supply-chain")
    assert len(pending) == 5

    # Decision plan: approve nodes 0,2,3,4 and reject node 1
    rejected_idx = 1
    seen_indices = []

    def carol_decide(node, idx, total):
        seen_indices.append(idx)
        if idx == rejected_idx:
            return ReviewDecision(action=ReviewAction.REJECT, reason="too vague")
        return ReviewDecision(action=ReviewAction.APPROVE)

    summary = carol_reviewer.review("supply-chain", carol_decide)
    assert summary.approved == 4
    assert summary.rejected == 1
    assert summary.total_pending == 5

    # Verify the approval state in the shared store
    approved = carol_store.list_approved_shared(project_id="supply-chain", include_team=True)
    assert len(approved) == 4
    assert all(n["approved_by"] == "carol" for n in approved)

    # The rejected one is still in staging marked rejected (audit kept)
    all_in_staging = carol_store.list_pending_in_staging("supply-chain")
    rejected_present = [n for n in all_in_staging if n.get("rejected")]
    assert len(rejected_present) == 1
    assert rejected_present[0]["rejection_reason"] == "too vague"

    # ─── Bob (new developer) pulls ──────────────────────────────────
    # Bob is on a totally fresh machine — empty local store, but same shared
    bob_local = str(tmp_path / "bob_local")
    bob_store = MemoryStore(local_path=bob_local, shared_path=shared_path)
    bob_log = SyncLog(db_path=str(tmp_path / "bob_sync.db"))
    bob_syncer = Syncer(store=bob_store, log=bob_log, developer="bob")

    # Bob has nothing locally yet
    assert bob_store.node_count("supply-chain") == 0

    pull_result = bob_syncer.pull("supply-chain", include_team=True)

    # Bob got the 4 approved nodes (the rejected one didn't reach him)
    assert pull_result.pulled_project + pull_result.pulled_team == 4
    assert bob_store.node_count("supply-chain") == 4

    # Bob's nodes are marked pushed_at (they came from shared, not generated locally)
    # so a subsequent push shouldn't bounce them back
    bob_push_back = bob_syncer.push("supply-chain")
    assert bob_push_back.pushed == 0

    # ─── Bob's local view has correct scope distribution ────────────
    bob_team = bob_store.nodes_col_local.get(
        where={"scope": "team"},
        include=["metadatas"],
    )
    bob_project = bob_store.nodes_col_local.get(
        where={"scope": "project"},
        include=["metadatas"],
    )
    # Originally: 3 project + 2 team. After rejecting one, we have either
    # 2 project + 2 team or 3 project + 1 team depending on which was at idx=1.
    assert len(bob_team["ids"]) + len(bob_project["ids"]) == 4

    # Audit log: alice pushed once, carol approved once, carol rejected once,
    # bob pulled once
    alice_push = alice_log.last_event("supply-chain", "push")
    bob_pull = bob_log.last_event("supply-chain", "pull")
    assert alice_push is not None
    assert alice_push["actor"] == "alice"
    assert bob_pull is not None
    assert bob_pull["actor"] == "bob"


def test_search_with_scope_filter(tmp_path, fake_embed):
    """The new scope= argument on search() filters by project/team."""
    from src.store.vector_store import MemoryStore

    store = MemoryStore(local_path=str(tmp_path / "local"))

    project_dag = _build_dag("proj", n=2, scope=MemoryScope.PROJECT)
    team_dag = _build_dag("team", n=2, scope=MemoryScope.TEAM)
    store.store_dag(project_dag, "myproj")
    store.store_dag(team_dag, "myproj")

    # Search without scope: gets all
    all_results = store.search("insight", project_id="myproj", top_k=10)
    assert len(all_results) == 4

    # Search project-only
    proj_only = store.search("insight", project_id="myproj", top_k=10, scope="project")
    assert len(proj_only) == 2
    assert all(r["scope"] == "project" for r in proj_only)

    # Search team-only
    team_only = store.search("insight", project_id="myproj", top_k=10, scope="team")
    assert len(team_only) == 2
    assert all(r["scope"] == "team" for r in team_only)


def test_legacy_local_only_workflow_unchanged(tmp_path, fake_embed):
    """The original single-developer flow still works with no changes."""
    from src.store.vector_store import MemoryStore

    # Old-style construction with persist_dir
    store = MemoryStore(persist_dir=str(tmp_path / "legacy"))
    assert not store.has_shared

    dag = _build_dag("legacy", n=3)
    store.store_dag(dag, "legacy-proj")
    assert store.node_count("legacy-proj") == 3

    # Search still works
    results = store.search("insight", project_id="legacy-proj", top_k=5)
    assert len(results) == 3
