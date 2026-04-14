#!/usr/bin/env python3
"""Install cognitive memory skills into a project's .claude/commands/ directory.

Usage:
    python scripts/install_skills.py /path/to/your/project --project my-project

This copies the skill templates from skills/ into the target project's
.claude/commands/ directory, configuring them with the correct paths and
project ID. It also creates the necessary environment wrapper so the
skills can locate the cmm Python environment and store.
"""
import argparse
import shutil
import sys
from pathlib import Path

CMM_ROOT = Path(__file__).parent.parent.resolve()
SKILLS_DIR = CMM_ROOT / "skills"
DEFAULT_STORE = CMM_ROOT / "data" / "memory_store"


def install(target_project: Path, project_id: str, store_dir: Path, python_path: Path):
    commands_dir = target_project / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    # Write an env file that the skills source for paths
    env_file = target_project / ".claude" / "cmm-env.sh"
    env_file.write_text(
        f'export CMM_PYTHON="{python_path}"\n'
        f'export CMM_PROJECT_ID="{project_id}"\n'
        f'export CMM_STORE_PATH="{store_dir}"\n'
        f'export CMM_ROOT="{CMM_ROOT}"\n'
        f'export CMM_ENV="{env_file}"\n'
        f'export PYTHONPATH="{CMM_ROOT}"\n'
    )

    # Copy and resolve each skill template
    for skill_file in SKILLS_DIR.glob("*.md"):
        content = skill_file.read_text()
        # Replace env var references with sourcing the env file + inline vars
        resolved = content.replace(
            '"$CMM_PYTHON" -m src.delivery.cli_query',
            f'cd "{CMM_ROOT}" && source "{env_file}" && "$CMM_PYTHON" -m src.delivery.cli_query',
        )
        dest = commands_dir / skill_file.name
        dest.write_text(resolved)
        print(f"  Installed: {dest}")

    # Remove MCP config if it only had cognitive-memory
    mcp_json = target_project / ".mcp.json"
    if mcp_json.exists():
        import json
        try:
            config = json.loads(mcp_json.read_text())
            servers = config.get("mcpServers", {})
            if "cognitive-memory" in servers:
                del servers["cognitive-memory"]
                mcp_json.write_text(json.dumps(config, indent=2) + "\n")
                print(f"  Removed cognitive-memory from {mcp_json}")
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"\nDone! Skills installed to {commands_dir}")
    print(f"Env config written to {env_file}")
    print(f"\nAvailable commands:")
    print(f"  /cognitive-profile    — full project knowledge")
    print(f"  /pitfalls             — known pitfalls ranked by severity")
    print(f"  /search-memory <q>    — semantic search over past reasoning")
    print(f"  /diagnose <problem>   — find proven debugging strategies")


def main():
    parser = argparse.ArgumentParser(description="Install cmm skills into a project")
    parser.add_argument("target", type=Path, help="Target project directory")
    parser.add_argument("--project", "-p", required=True, help="Project ID for this project")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=DEFAULT_STORE,
        help=f"Memory store directory (default: {DEFAULT_STORE})",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=CMM_ROOT / ".venv" / "bin" / "python",
        help="Path to the Python interpreter with cmm dependencies",
    )
    args = parser.parse_args()

    if not args.target.is_dir():
        print(f"Error: {args.target} is not a directory", file=sys.stderr)
        sys.exit(1)

    if not args.python.exists():
        print(f"Warning: Python path {args.python} does not exist", file=sys.stderr)

    install(args.target, args.project, args.store_dir, args.python)


if __name__ == "__main__":
    main()
