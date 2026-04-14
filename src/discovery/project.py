"""Project discovery via .cmm/ folder.

The .cmm/ folder in a project root is the self-discovery mechanism.
It tells any session: "this project has cognitive memory — here's how to find it."

Structure:
    .cmm/
    ├── manifest.json      # project identity + metadata
    ├── llms.txt           # structured metadata for agent consumption
    ├── cached_profile.md  # latest cognitive profile in markdown
    └── config.json        # memory system settings

The project ID is derived from the repo name + a hash of the README content,
so it's stable across machines and doesn't need to be hardcoded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_COGNITIVE_DIR = ".cmm"
_MANIFEST_FILE = "manifest.json"
_CONFIG_FILE = "config.json"
_CACHED_PROFILE_FILE = "cached_profile.md"
_LLMS_TXT_FILE = "llms.txt"

_DEFAULT_CONFIG = {
    "auto_ingest": True,
    "auto_retrieve": True,
    "similarity_threshold": 0.85,
    "max_search_results": 5,
    "warm_extraction_only": True,
    # Phase 3-6: shared-store sync
    "shared_store_path": None,
    "local_store_path": None,        # null = use default ~/.cognitive-memory/store
    "mode": "local",                  # "local" or "shared"
    "auto_push": False,
    "context_fill_ratio": 0.45,
    "developer_name": None,
    "team_id": None,
}


def _get_repo_name(project_dir: Path) -> str:
    """Get the git repo name, or fall back to directory name."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(project_dir),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL: git@github.com:user/repo.git → repo
            name = url.rstrip("/").rsplit("/", 1)[-1]
            return name.removesuffix(".git")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return project_dir.name


def _read_readme(project_dir: Path) -> str:
    """Read README content for hashing. Returns empty string if not found."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = project_dir / name
        if readme.exists():
            try:
                return readme.read_text(errors="replace")[:5000]
            except OSError:
                pass
    return ""


def generate_project_id(project_dir: Path) -> str:
    """Generate a stable project ID from repo name + README hash.

    Format: <repo-name>-<short-hash>
    Example: supply-chain-tracker-a3f2b1
    """
    repo_name = _get_repo_name(project_dir)
    readme_content = _read_readme(project_dir)
    # Hash the readme content (or empty string) for uniqueness
    content_hash = hashlib.sha256(readme_content.encode()).hexdigest()[:6]
    # Clean the repo name
    clean_name = repo_name.lower().replace(" ", "-").replace("_", "-")
    return f"{clean_name}-{content_hash}"


@dataclass
class CognitiveProject:
    """Represents a project with cognitive memory initialized."""

    project_dir: Path
    project_id: str
    name: str
    description: str = ""
    repo_path: str = ""
    created_at: str = ""
    last_session: str = ""
    session_count: int = 0
    config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_CONFIG))

    @property
    def cognitive_dir(self) -> Path:
        return self.project_dir / _COGNITIVE_DIR

    @property
    def manifest_path(self) -> Path:
        return self.cognitive_dir / _MANIFEST_FILE

    @property
    def config_path(self) -> Path:
        return self.cognitive_dir / _CONFIG_FILE

    @property
    def cached_profile_path(self) -> Path:
        return self.cognitive_dir / _CACHED_PROFILE_FILE

    @property
    def llms_txt_path(self) -> Path:
        return self.cognitive_dir / _LLMS_TXT_FILE

    # ── Initialization ────────────────────────────────────────────

    @classmethod
    def init(cls, project_dir: Path, store_path: str | None = None) -> "CognitiveProject":
        """Initialize a .cmm/ folder in the project directory."""
        project_dir = project_dir.resolve()
        project_id = generate_project_id(project_dir)
        repo_name = _get_repo_name(project_dir)

        # Read description from README first line
        readme = _read_readme(project_dir)
        description = ""
        for line in readme.split("\n"):
            stripped = line.strip().lstrip("#").strip()
            if stripped and not stripped.startswith("[") and not stripped.startswith("!"):
                description = stripped[:200]
                break

        now = datetime.now(timezone.utc).isoformat()

        proj = cls(
            project_dir=project_dir,
            project_id=project_id,
            name=repo_name,
            description=description,
            repo_path=str(project_dir),
            created_at=now,
            last_session="",
            session_count=0,
        )

        # Create .cmm/ directory
        proj.cognitive_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest
        proj.save_manifest()

        # Write default config
        if store_path:
            proj.config["store_path"] = store_path
        proj.save_config()

        # Write initial cached_profile.md
        proj.cached_profile_path.write_text(
            f"# Cognitive Profile: {repo_name}\n\n"
            f"*No sessions ingested yet. Run `cmm ingest` or start a coding session.*\n"
        )

        return proj

    # ── Persistence ───────────────────────────────────────────────

    def save_manifest(self):
        """Write manifest.json."""
        data = {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "repo_path": self.repo_path,
            "created_at": self.created_at,
            "last_session": self.last_session,
            "session_count": self.session_count,
        }
        self.manifest_path.write_text(json.dumps(data, indent=2) + "\n")

    def save_config(self):
        """Write config.json."""
        self.config_path.write_text(json.dumps(self.config, indent=2) + "\n")

    def update_session(self, session_id: str = ""):
        """Record that a session was ingested."""
        self.last_session = session_id or datetime.now(timezone.utc).isoformat()
        self.session_count += 1
        self.save_manifest()

    def update_cached_profile(self, profile_markdown: str):
        """Write the latest profile as markdown for fast loading."""
        self.cached_profile_path.write_text(profile_markdown)

    # ── Loading ───────────────────────────────────────────────────

    @classmethod
    def load(cls, project_dir: Path) -> "CognitiveProject":
        """Load an existing .cmm/ folder."""
        project_dir = project_dir.resolve()
        cognitive_dir = project_dir / _COGNITIVE_DIR
        manifest_path = cognitive_dir / _MANIFEST_FILE

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No .cmm/ folder found in {project_dir}. Run 'cmm init' first."
            )

        manifest = json.loads(manifest_path.read_text())
        config_path = cognitive_dir / _CONFIG_FILE
        config = json.loads(config_path.read_text()) if config_path.exists() else dict(_DEFAULT_CONFIG)

        return cls(
            project_dir=project_dir,
            project_id=manifest["project_id"],
            name=manifest.get("name", ""),
            description=manifest.get("description", ""),
            repo_path=manifest.get("repo_path", str(project_dir)),
            created_at=manifest.get("created_at", ""),
            last_session=manifest.get("last_session", ""),
            session_count=manifest.get("session_count", 0),
            config=config,
        )


def discover_project(start_dir: Path | None = None) -> CognitiveProject | None:
    """Walk up from start_dir looking for a .cmm/ folder.

    Returns CognitiveProject if found, None otherwise.
    This is the main entry point for automatic discovery.
    """
    current = (start_dir or Path.cwd()).resolve()

    # Walk up to filesystem root
    for _ in range(50):  # safety limit
        cognitive_dir = current / _COGNITIVE_DIR
        if (cognitive_dir / _MANIFEST_FILE).exists():
            return CognitiveProject.load(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None
