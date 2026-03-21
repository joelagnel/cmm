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
