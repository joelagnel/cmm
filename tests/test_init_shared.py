"""Tests for the shared-mode init / onboarding flow and classify command."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.schemas.reasoning import (
    MemoryScope,
    NodeType,
    ReasoningDAG,
    ReasoningNode,
)


@pytest.fixture
def fake_embed():
    def _embed(self, texts):
        return [[float(i) / 10 + 0.01 * j for j in range(8)] for i in range(len(texts))]
    with patch("src.store.vector_store.MemoryStore.embed", _embed):
        yield


# ── Local-only init backwards-compatible ────────────────────────────────


def test_init_local_only(tmp_path):
    from src.cli import main as cli

    runner = CliRunner()
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# myproj\nA test project")

    result = runner.invoke(cli, ["init", str(project_dir)])
    assert result.exit_code == 0

    cfg = json.loads((project_dir / ".cmm" / "config.json").read_text())
    assert cfg["mode"] == "local"
    assert cfg["shared_store_path"] is None


def test_init_writes_developer_and_team_id(tmp_path):
    from src.cli import main as cli

    runner = CliRunner()
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# myproj")

    result = runner.invoke(cli, [
        "init", str(project_dir),
        "--developer", "alice",
        "--team-id", "logistics",
    ])
    assert result.exit_code == 0

    cfg = json.loads((project_dir / ".cmm" / "config.json").read_text())
    assert cfg["developer_name"] == "alice"
    assert cfg["team_id"] == "logistics"


# ── Shared-mode init ────────────────────────────────────────────────────


def test_init_shared_writes_shared_path(tmp_path, fake_embed):
    from src.cli import main as cli

    runner = CliRunner()
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# myproj")

    shared_path = str(tmp_path / "shared")
    local_path = str(tmp_path / "local")

    result = runner.invoke(cli, [
        "init", str(project_dir),
        "--store-dir", local_path,
        "--shared", shared_path,
        "--developer", "alice",
    ])
    assert result.exit_code == 0, result.output

    cfg = json.loads((project_dir / ".cmm" / "config.json").read_text())
    assert cfg["mode"] == "shared"
    assert cfg["shared_store_path"] == shared_path
    assert cfg["developer_name"] == "alice"


def test_init_shared_pulls_existing_team_memories(tmp_path, fake_embed):
    """Onboarding flow: cmm init --shared should run pull automatically."""
    from src.cli import main as cli
    from src.store.vector_store import MemoryStore

    # First, alice seeds the shared store with an approved team-scope node
    shared_path = str(tmp_path / "shared")
    alice_local = str(tmp_path / "alice_local")

    alice_store = MemoryStore(local_path=alice_local, shared_path=shared_path)
    dag = ReasoningDAG(
        session_id="s1",
        nodes=[ReasoningNode(
            node_id="n0",
            node_type=NodeType.DISCOVERY,
            summary="Pydantic v2 ConfigDict migration",
            evidence="...",
            message_range=(0, 1),
            confidence=0.9,
            scope=MemoryScope.TEAM,
        )],
        edges=[],
    )
    alice_store.store_dag(dag, "alice-proj")
    unpushed = alice_store.get_unpushed_nodes("alice-proj")
    alice_store.stage_to_shared(unpushed, developer="alice")
    pending_ids = [p["id"] for p in alice_store.list_pending_in_staging("alice-proj")]
    alice_store.promote_from_staging(pending_ids, approver="reviewer")

    # Now bob runs cmm init --shared on a different project
    bob_project = tmp_path / "bob_proj"
    bob_project.mkdir()
    (bob_project / "README.md").write_text("# bob_proj")
    bob_local = str(tmp_path / "bob_local")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "init", str(bob_project),
        "--store-dir", bob_local,
        "--shared", shared_path,
        "--developer", "bob",
    ])
    assert result.exit_code == 0, result.output

    # Bob should have pulled the team-scope node into his local store
    bob_store = MemoryStore(local_path=bob_local, shared_path=shared_path)
    team_nodes = bob_store.nodes_col_local.get(
        where={"scope": "team"},
        include=["documents"],
    )
    assert len(team_nodes["ids"]) == 1


def test_init_idempotent(tmp_path):
    """Running init twice doesn't crash and updates config if --shared added."""
    from src.cli import main as cli

    runner = CliRunner()
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# myproj")

    runner.invoke(cli, ["init", str(project_dir)])
    # Re-run with --developer
    result = runner.invoke(cli, [
        "init", str(project_dir),
        "--developer", "alice",
    ])
    assert result.exit_code == 0
    cfg = json.loads((project_dir / ".cmm" / "config.json").read_text())
    assert cfg["developer_name"] == "alice"


# ── classify command ───────────────────────────────────────────────────


def test_classify_changes_scope(tmp_path, fake_embed):
    from src.cli import main as cli
    from src.store.vector_store import MemoryStore

    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# myproj")

    runner = CliRunner()
    runner.invoke(cli, ["init", str(project_dir), "--store-dir", str(tmp_path / "store")])

    # Manually seed a node
    store = MemoryStore(local_path=str(tmp_path / "store"))
    from src.discovery.project import CognitiveProject
    proj = CognitiveProject.load(project_dir)
    dag = ReasoningDAG(
        session_id="s1",
        nodes=[ReasoningNode(
            node_id="n0",
            node_type=NodeType.HYPOTHESIS,
            summary="some local insight",
            evidence="...",
            message_range=(0, 1),
            confidence=0.7,
            scope=MemoryScope.PROJECT,
        )],
        edges=[],
    )
    store.store_dag(dag, proj.project_id)

    # Find the actual stored ID
    all_ids = store.nodes_col_local.get(include=[])["ids"]
    assert len(all_ids) == 1
    node_id = all_ids[0]

    result = runner.invoke(cli, [
        "classify", node_id,
        "--scope", "team",
        "--target", str(project_dir),
        "--project", proj.project_id,
    ])
    assert result.exit_code == 0, result.output

    # Verify the scope changed
    meta = store.nodes_col_local.get(ids=[node_id], include=["metadatas"])
    assert meta["metadatas"][0]["scope"] == "team"


def test_default_config_has_new_fields():
    from src.discovery.project import _DEFAULT_CONFIG
    expected_keys = {
        "shared_store_path", "local_store_path", "mode",
        "auto_push", "context_fill_ratio", "developer_name", "team_id",
    }
    assert expected_keys.issubset(_DEFAULT_CONFIG.keys())
    assert _DEFAULT_CONFIG["mode"] == "local"
    assert _DEFAULT_CONFIG["context_fill_ratio"] == 0.45
