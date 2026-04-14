"""Tests for project discovery, .cmm/ folder, and hooks — Phase 5 exit criteria."""
import json
import pytest
from pathlib import Path
from datetime import datetime

from src.discovery.project import (
    CognitiveProject,
    discover_project,
    generate_project_id,
)
from src.discovery.llms_txt import generate_llms_txt, _detect_stack
from src.discovery.hooks import session_start_hook, session_stop_hook


# ── Project ID Generation ─────────────────────────────────────────


def test_generate_project_id_from_dir_name(tmp_path):
    """Project ID is generated from directory name when no git remote."""
    project = tmp_path / "my-cool-project"
    project.mkdir()
    pid = generate_project_id(project)
    assert pid.startswith("my-cool-project-")
    assert len(pid.split("-")[-1]) == 6  # 6-char hash


def test_generate_project_id_stable(tmp_path):
    """Same directory produces the same project ID."""
    project = tmp_path / "my-project"
    project.mkdir()
    pid1 = generate_project_id(project)
    pid2 = generate_project_id(project)
    assert pid1 == pid2


def test_generate_project_id_changes_with_readme(tmp_path):
    """Project ID changes when README content changes."""
    project = tmp_path / "my-project"
    project.mkdir()

    pid1 = generate_project_id(project)

    (project / "README.md").write_text("# My Project\nThis is a test project.")
    pid2 = generate_project_id(project)

    # Name prefix same, hash differs
    assert pid1.rsplit("-", 1)[0] == pid2.rsplit("-", 1)[0]
    assert pid1 != pid2


# ── CognitiveProject Init ─────────────────────────────────────────


def test_init_creates_cognitive_dir(tmp_path):
    """cmm init creates .cmm/ with all required files."""
    project = tmp_path / "test-project"
    project.mkdir()
    (project / "README.md").write_text("# Test Project\nA test.")

    proj = CognitiveProject.init(project)

    assert (project / ".cmm").is_dir()
    assert (project / ".cmm" / "manifest.json").exists()
    assert (project / ".cmm" / "config.json").exists()
    assert (project / ".cmm" / "cached_profile.md").exists()


def test_init_manifest_content(tmp_path):
    """Manifest contains correct project identity."""
    project = tmp_path / "test-project"
    project.mkdir()
    (project / "README.md").write_text("# My App\nSome description here.")

    proj = CognitiveProject.init(project)
    manifest = json.loads((project / ".cmm" / "manifest.json").read_text())

    assert manifest["name"] == "test-project"
    assert manifest["project_id"] == proj.project_id
    assert manifest["description"] == "My App"
    assert manifest["session_count"] == 0
    assert manifest["repo_path"] == str(project)


def test_init_config_defaults(tmp_path):
    """Config has sensible defaults."""
    project = tmp_path / "test-project"
    project.mkdir()

    proj = CognitiveProject.init(project)
    config = json.loads((project / ".cmm" / "config.json").read_text())

    assert config["auto_ingest"] is True
    assert config["auto_retrieve"] is True
    assert config["similarity_threshold"] == 0.85
    assert config["max_search_results"] == 5


def test_init_with_store_path(tmp_path):
    """Custom store path is saved in config."""
    project = tmp_path / "test-project"
    project.mkdir()

    proj = CognitiveProject.init(project, store_path="/custom/store")
    config = json.loads((project / ".cmm" / "config.json").read_text())

    assert config["store_path"] == "/custom/store"


# ── CognitiveProject Load ─────────────────────────────────────────


def test_load_existing_project(tmp_path):
    """Loading a previously initialized project recovers all data."""
    project = tmp_path / "test-project"
    project.mkdir()
    (project / "README.md").write_text("# Test\nDesc.")

    original = CognitiveProject.init(project)
    original.update_session("session-123")

    loaded = CognitiveProject.load(project)
    assert loaded.project_id == original.project_id
    assert loaded.name == original.name
    assert loaded.session_count == 1
    assert loaded.last_session == "session-123"


def test_load_nonexistent_raises(tmp_path):
    """Loading from a directory without .cmm/ raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No .cmm/ folder"):
        CognitiveProject.load(tmp_path)


# ── discover_project ──────────────────────────────────────────────


def test_discover_project_finds_in_current_dir(tmp_path):
    """discover_project finds .cmm/ in the given directory."""
    project = tmp_path / "my-project"
    project.mkdir()
    CognitiveProject.init(project)

    found = discover_project(project)
    assert found is not None
    assert found.name == "my-project"


def test_discover_project_walks_up(tmp_path):
    """discover_project walks up from a subdirectory."""
    project = tmp_path / "my-project"
    project.mkdir()
    CognitiveProject.init(project)

    subdir = project / "src" / "deep" / "nested"
    subdir.mkdir(parents=True)

    found = discover_project(subdir)
    assert found is not None
    assert found.project_id == CognitiveProject.load(project).project_id


def test_discover_project_returns_none_when_missing(tmp_path):
    """discover_project returns None when no .cmm/ exists."""
    found = discover_project(tmp_path)
    assert found is None


# ── update_session ────────────────────────────────────────────────


def test_update_session_increments_count(tmp_path):
    """update_session increments session_count and saves."""
    project = tmp_path / "test-project"
    project.mkdir()
    proj = CognitiveProject.init(project)

    proj.update_session("sess-1")
    assert proj.session_count == 1

    proj.update_session("sess-2")
    assert proj.session_count == 2

    # Verify persistence
    reloaded = CognitiveProject.load(project)
    assert reloaded.session_count == 2
    assert reloaded.last_session == "sess-2"


# ── llms.txt Generation ──────────────────────────────────────────


def test_generate_llms_txt_no_profile():
    """llms.txt generates valid content without a profile."""
    content = generate_llms_txt(
        project_name="my-project",
        project_description="A test project",
        profile=None,
    )
    assert "# my-project" in content
    assert "A test project" in content
    assert "sessions_analyzed: 0" in content
    assert "Retrieval Hints" in content


def test_generate_llms_txt_with_profile():
    """llms.txt includes profile data when available."""
    from src.schemas.memory import CognitiveProfile, Pitfall, ArchitecturalInsight

    profile = CognitiveProfile(
        project_id="test",
        last_updated=datetime.now(),
        architectural_insights=[
            ArchitecturalInsight(component="API", insight="Uses FastAPI with async routes", confidence=0.9),
        ],
        pitfalls=[
            Pitfall(description="DB engine created at import time", severity="high", resolution_strategy="Use lazy init"),
        ],
        session_count=5,
    )

    content = generate_llms_txt(
        project_name="my-project",
        project_description="A FastAPI backend",
        profile=profile,
    )

    assert "sessions_analyzed: 5" in content
    assert "DB engine created at import time" in content
    assert "Uses FastAPI with async routes" in content
    assert "[HIGH]" in content


def test_detect_stack(tmp_path):
    """Stack detection identifies common technologies."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\ndependencies=['fastapi']")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12")

    stack = _detect_stack(tmp_path)
    assert "Python" in stack
    assert "FastAPI" in stack
    assert "Docker" in stack


# ── Hooks ─────────────────────────────────────────────────────────


def test_session_start_hook_no_cognitive_dir(tmp_path):
    """Start hook returns helpful message when no .cmm/ exists."""
    output = session_start_hook(tmp_path)
    assert "No cognitive memory found" in output
    assert "cmm init" in output


def test_session_start_hook_with_cognitive_dir(tmp_path):
    """Start hook returns cached profile content when .cmm/ exists."""
    project = tmp_path / "test-project"
    project.mkdir()
    proj = CognitiveProject.init(project)

    # Write a meaningful cached profile
    proj.update_cached_profile(
        "# Cognitive Profile: test-project\n\n"
        "## Architectural Insights\n- Uses async database connections\n"
    )

    output = session_start_hook(project)
    assert "Architectural Insights" in output
    assert "async database connections" in output


def test_session_stop_hook_no_cognitive_dir(tmp_path):
    """Stop hook skips when no .cmm/ exists."""
    result = session_stop_hook(tmp_path)
    assert result["status"] == "skipped"
    assert "no .cmm/" in result["reason"]


def test_session_stop_hook_auto_ingest_disabled(tmp_path):
    """Stop hook respects auto_ingest=false in config."""
    project = tmp_path / "test-project"
    project.mkdir()
    proj = CognitiveProject.init(project)
    proj.config["auto_ingest"] = False
    proj.save_config()

    result = session_stop_hook(project)
    assert result["status"] == "skipped"
    assert "auto_ingest disabled" in result["reason"]
