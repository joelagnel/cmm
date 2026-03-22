"""cmm — cognitive memory CLI entry point."""
import click


@click.group()
def main():
    """Cognitive Memory Manager — persistent cross-platform memory for AI coding agents."""


@main.command()
@click.argument("sessions", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--project", "-p", required=True, help="Project ID")
@click.option("--store-dir", default=None, help="Memory store directory (default: ~/.cognitive-memory/store)")
@click.option("--build-profile", is_flag=True, help="Build cognitive profile after ingestion")
@click.option("--no-llm", is_flag=True, help="Use heuristic extraction (no API calls)")
def ingest(sessions, project, store_dir, build_profile, no_llm):
    """Ingest session files into the memory store."""
    import asyncio
    import sys
    from pathlib import Path

    # Defer import to avoid loading heavy deps at CLI parse time
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.ingest import main as ingest_main

    # Re-invoke the ingest script's async main with the same arguments
    import sys
    sys.argv = ["ingest"] + list(sessions) + ["-p", project]
    if store_dir:
        sys.argv += ["--store-dir", store_dir]
    if build_profile:
        sys.argv.append("--build-profile")
    if no_llm:
        sys.argv.append("--no-llm")
    asyncio.run(ingest_main())


@main.command()
@click.option("--project", "-p", required=True, help="Project ID")
@click.option("--store-dir", default=None, help="Memory store directory")
def status(project, store_dir):
    """Show memory store status for a project."""
    import sys
    sys.argv = ["ingest", "--status", "-p", project]
    if store_dir:
        sys.argv += ["--store-dir", store_dir]

    import asyncio
    from scripts.ingest import main as ingest_main
    asyncio.run(ingest_main())


@main.command()
@click.option("--store-path", envvar="CMM_STORE_PATH",
              default=None, help="Memory store path (or set CMM_STORE_PATH)")
@click.option("--project", envvar="CMM_PROJECT_ID",
              default=None, help="Default project ID (or set CMM_PROJECT_ID)")
def serve(store_path, project):
    """Start the MCP server (stdio transport)."""
    import os
    if store_path:
        os.environ["CMM_STORE_PATH"] = store_path
    if project:
        os.environ["CMM_PROJECT_ID"] = project

    from src.delivery.mcp_server import run
    run()


@main.command()
@click.option("--project", "-p", required=True, help="Project ID")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--output", "-o", default="output/", help="Output directory")
@click.option("--format", "-f", "fmt", default="html",
              type=click.Choice(["html", "mermaid", "json", "all"]),
              help="Output format (default: html)")
def visualize(project, store_dir, output, fmt):
    """Generate an interactive DAG visualization."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.visualize_dag import main as viz_main
    sys.argv = ["visualize", "-p", project, "-o", output, "-f", fmt]
    if store_dir:
        sys.argv += ["--store", store_dir]
    viz_main()


@main.command()
@click.option("--projects-dir", default=None, help="Claude Code projects directory (default: ~/.claude/projects/)")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--poll-interval", type=float, default=10.0, help="Seconds between polls (default: 10)")
@click.option("--min-age", type=float, default=30.0, help="Min file age before processing (default: 30s)")
@click.option("--no-auto-ingest", is_flag=True, help="Only detect, don't auto-ingest")
def watch(projects_dir, store_dir, poll_interval, min_age, no_auto_ingest):
    """Watch for new Claude Code sessions and auto-ingest them."""
    import asyncio
    from pathlib import Path
    from src.ingestion.watcher import SessionWatcher

    store_path = store_dir or str(Path(__file__).parent.parent / "data" / "memory_store")

    watcher = SessionWatcher(
        watch_dir=projects_dir,
        store_path=store_path,
        poll_interval=poll_interval,
        min_file_age=min_age,
        auto_ingest=not no_auto_ingest,
    )
    asyncio.run(watcher.watch())


@main.command()
@click.option("--project", "-p", default=None, help="Project ID (or --all)")
@click.option("--all", "all_projects", is_flag=True, help="Consolidate all projects")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--upgrade", is_flag=True, help="Re-extract warm-tier nodes with LLM")
@click.option("--profiles-only", is_flag=True, help="Only rebuild profiles")
@click.option("--dry-run", is_flag=True, help="Show what would be processed")
def consolidate(project, all_projects, store_dir, upgrade, profiles_only, dry_run):
    """Run batch consolidation — rebuild profiles and optionally upgrade warm nodes."""
    import asyncio
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.batch_consolidate import main as consolidate_main

    argv = ["consolidate"]
    if project:
        argv += ["-p", project]
    if all_projects:
        argv.append("--all")
    if store_dir:
        argv += ["--store-dir", store_dir]
    if upgrade:
        argv.append("--upgrade")
    if profiles_only:
        argv.append("--profiles-only")
    if dry_run:
        argv.append("--dry-run")

    sys.argv = argv
    asyncio.run(consolidate_main())


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("--project", "-p", required=True, help="Project ID")
@click.option("--store-dir", default=None, help="Memory store directory")
@click.option("--python", default=None, help="Path to Python with cmm deps")
def install(target, project, store_dir, python):
    """Install cognitive memory skills into a project's .claude/commands/."""
    from pathlib import Path
    from scripts.install_skills import install as do_install, CMM_ROOT, DEFAULT_STORE

    python_path = Path(python) if python else CMM_ROOT / ".venv" / "bin" / "python"
    store_path = Path(store_dir) if store_dir else DEFAULT_STORE
    do_install(Path(target), project, store_path, python_path)
