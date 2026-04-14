#!/usr/bin/env python3
"""Cold-tier batch consolidation — full reprocessing and profile rebuilding.

Runs periodically (e.g., nightly via cron) or on-demand. Performs:
  1. Finds all sessions that only had warm-tier processing
  2. Runs full LLM-based DAG extraction on them (if API key available)
  3. Deduplicates against existing memory
  4. Rebuilds cognitive profiles for all affected projects

Usage:
    # Consolidate a specific project
    python scripts/batch_consolidate.py -p supply-chain

    # Consolidate all projects in the store
    python scripts/batch_consolidate.py --all

    # Dry run — show what would be processed
    python scripts/batch_consolidate.py --all --dry-run

    # Re-extract warm-tier nodes with LLM (upgrade quality)
    python scripts/batch_consolidate.py -p supply-chain --upgrade

    # Just rebuild profiles (no re-extraction)
    python scripts/batch_consolidate.py -p supply-chain --profiles-only
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.store import MemoryStore
from src.compression import SemanticDeduplicator, ProfileBuilder

console = Console()
logger = logging.getLogger(__name__)

_DEFAULT_STORE = Path(__file__).parent.parent / "data" / "memory_store"


def _find_warm_nodes(store: MemoryStore, project_id: str) -> list[dict]:
    """Find nodes that were extracted by the warm tier (low confidence, heuristic evidence)."""
    all_nodes = store.get_all_nodes(project_id)
    warm_nodes = []
    for node in all_nodes:
        meta = node.get("metadata", {})
        # Warm-tier nodes have node_ids starting with "warm-" or "hnode-"
        # and typically have confidence <= 0.5
        node_id = meta.get("node_id", "")
        confidence = float(meta.get("confidence", 0))
        if (node_id.startswith("warm-") or node_id.startswith("hnode-")) and confidence <= 0.5:
            warm_nodes.append(node)
    return warm_nodes


def _find_session_files(watcher_state_path: Path) -> dict[str, list[Path]]:
    """Read the watcher state to find ingested session files by project."""
    if not watcher_state_path.exists():
        return {}
    try:
        data = json.loads(watcher_state_path.read_text())
        sessions_by_project: dict[str, list[Path]] = {}
        for entry in data.get("processed", []):
            path = Path(entry["path"])
            # Derive project from the parent directory name
            project_dir = path.parent.name
            sessions_by_project.setdefault(project_dir, []).append(path)
        return sessions_by_project
    except (json.JSONDecodeError, KeyError):
        return {}


async def upgrade_warm_nodes(
    store: MemoryStore,
    project_id: str,
) -> dict:
    """Re-extract warm-tier nodes using full LLM extraction."""
    from src.ingestion import ClaudeCodeParser
    from src.extraction import DAGBuilder

    warm_nodes = _find_warm_nodes(store, project_id)
    if not warm_nodes:
        return {"upgraded": 0, "skipped": 0}

    # Group warm nodes by session_id to find source files
    sessions_to_reprocess = set()
    for node in warm_nodes:
        sid = node.get("metadata", {}).get("session_id", "")
        if sid:
            sessions_to_reprocess.add(sid)

    # Try to find the original session files via watcher state
    state_file = Path(store._persist_dir) / ".cmm_watcher_state.json"
    session_map = _find_session_files(state_file)

    upgraded = 0
    skipped = 0

    # Also check ~/.claude/projects/ for session files
    claude_projects = Path(os.path.expanduser("~/.claude/projects"))
    if claude_projects.exists():
        for project_dir in claude_projects.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                # Check if this session ID matches one we need to reprocess
                session_id = jsonl_file.stem
                if session_id in sessions_to_reprocess:
                    console.print(f"  Re-extracting [cyan]{jsonl_file.name}[/cyan]...")
                    try:
                        parser = ClaudeCodeParser()
                        session = parser.parse_file(jsonl_file)

                        builder = DAGBuilder()
                        dag = await builder.build(session)

                        dedup = SemanticDeduplicator(store)
                        result = dedup.deduplicate(dag.nodes, project_id, session.session_id)

                        from src.schemas.reasoning import ReasoningDAG
                        dag_to_store = ReasoningDAG(
                            session_id=dag.session_id,
                            nodes=result.stored,
                            edges=dag.edges,
                            pivot_nodes=dag.pivot_nodes,
                            noise_ratio=dag.noise_ratio,
                        )
                        store.store_dag(dag_to_store, project_id)
                        upgraded += 1
                        sessions_to_reprocess.discard(session_id)
                    except Exception as e:
                        console.print(f"    [red]Failed: {e}[/red]")
                        skipped += 1

    skipped += len(sessions_to_reprocess)  # Sessions we couldn't find files for
    return {"upgraded": upgraded, "skipped": skipped}


async def rebuild_profile(store: MemoryStore, project_id: str) -> dict:
    """Rebuild the cognitive profile for a project."""
    node_count = store.node_count(project_id)
    if node_count == 0:
        return {"status": "empty", "node_count": 0}

    builder = ProfileBuilder()
    profile = await builder.build_profile(project_id, store)

    return {
        "status": "built",
        "node_count": node_count,
        "insights": len(profile.architectural_insights),
        "pitfalls": len(profile.pitfalls),
        "strategies": len(profile.diagnostic_strategies),
        "patterns": len(profile.key_patterns),
    }


async def consolidate_project(
    store: MemoryStore,
    project_id: str,
    has_llm: bool,
    upgrade: bool = False,
    profiles_only: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run full consolidation for a single project."""
    result = {"project_id": project_id}

    # Step 1: Count current state
    node_count = store.node_count(project_id)
    warm_nodes = _find_warm_nodes(store, project_id)
    result["total_nodes"] = node_count
    result["warm_nodes"] = len(warm_nodes)

    if dry_run:
        result["action"] = "dry_run"
        return result

    # Step 2: Upgrade warm nodes if requested and LLM credentials available
    if upgrade and has_llm and not profiles_only:
        console.print(f"\n[bold]Upgrading warm-tier nodes for {project_id}...[/bold]")
        upgrade_result = await upgrade_warm_nodes(store, project_id)
        result["upgrade"] = upgrade_result
    elif upgrade and not has_llm:
        console.print("[yellow]  No LLM credentials set -- skipping warm-tier upgrade[/yellow]")

    # Step 3: Rebuild profile
    if has_llm:
        console.print(f"[bold]Rebuilding cognitive profile for {project_id}...[/bold]")
        profile_result = await rebuild_profile(store, project_id)
        result["profile"] = profile_result
    else:
        console.print("[yellow]  No LLM credentials set -- skipping profile rebuild[/yellow]")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Batch consolidation for cognitive memory")
    parser.add_argument("--project", "-p", help="Project ID to consolidate")
    parser.add_argument("--all", action="store_true", help="Consolidate all projects")
    parser.add_argument("--store-dir", default=str(_DEFAULT_STORE), help="Memory store directory")
    parser.add_argument("--upgrade", action="store_true",
                        help="Re-extract warm-tier nodes with full LLM extraction")
    parser.add_argument("--profiles-only", action="store_true",
                        help="Only rebuild profiles, skip re-extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without doing it")

    args = parser.parse_args()

    if not args.project and not args.all:
        console.print("[red]Specify --project/-p or --all[/red]")
        parser.print_help()
        sys.exit(1)

    has_llm_credentials = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
    )
    store = MemoryStore(persist_dir=args.store_dir)

    # Determine which projects to process
    if args.all:
        projects = store.list_projects()
        if not projects:
            console.print("[yellow]No projects found in the store.[/yellow]")
            return
    else:
        projects = [args.project]

    console.rule("[bold]Batch Consolidation")

    results_table = Table(title="Consolidation Results", show_header=True)
    results_table.add_column("Project", style="cyan")
    results_table.add_column("Nodes")
    results_table.add_column("Warm")
    results_table.add_column("Upgraded")
    results_table.add_column("Profile")

    for project_id in projects:
        console.print(f"\nProcessing [bold cyan]{project_id}[/bold cyan]...")
        result = await consolidate_project(
            store=store,
            project_id=project_id,
            has_llm=has_llm_credentials,
            upgrade=args.upgrade,
            profiles_only=args.profiles_only,
            dry_run=args.dry_run,
        )

        upgraded_str = "-"
        if "upgrade" in result:
            u = result["upgrade"]
            upgraded_str = f"{u['upgraded']} ({u['skipped']} skipped)"

        profile_str = "-"
        if "profile" in result:
            p = result["profile"]
            if p["status"] == "built":
                profile_str = f"{p['insights']}i/{p['pitfalls']}p/{p['strategies']}s"
            else:
                profile_str = p["status"]
        elif args.dry_run:
            profile_str = "dry run"

        results_table.add_row(
            project_id,
            str(result.get("total_nodes", 0)),
            str(result.get("warm_nodes", 0)),
            upgraded_str,
            profile_str,
        )

    console.print()
    console.print(results_table)
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
