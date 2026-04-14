#!/usr/bin/env python3
"""End-to-end pipeline: session.jsonl → NormalizedSession → ReasoningDAG.

Usage:
    python scripts/run_pipeline.py <session.jsonl>
    python scripts/run_pipeline.py fixtures/claude_code/session_detection_01.jsonl
    python scripts/run_pipeline.py fixtures/synthetic/debugging_with_pivot.jsonl --no-llm
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.ingestion import ClaudeCodeParser
from src.extraction import DAGBuilder
from src.schemas.reasoning import NodeType

console = Console()


def print_session_summary(session):
    from collections import Counter
    role_counts = Counter(m.role.value for m in session.messages)

    table = Table(title=f"Session: {session.session_id}", show_header=True)
    table.add_column("Attribute", style="cyan")
    table.add_column("Value")

    table.add_row("Platform", session.source_platform)
    table.add_row("Project", session.project_path or "unknown")
    table.add_row("Total messages", str(len(session.messages)))
    table.add_row("Started", str(session.started_at))
    table.add_row("Ended", str(session.ended_at))
    for role, count in sorted(role_counts.items()):
        table.add_row(f"  {role}", str(count))

    all_files = list({f for m in session.messages for f in m.files_modified})
    table.add_row("Files modified", str(len(all_files)))
    if all_files:
        table.add_row("  (sample)", all_files[0])

    console.print(table)


def print_dag_summary(dag):
    from collections import Counter
    type_counts = Counter(n.node_type.value for n in dag.nodes)

    table = Table(title=f"Reasoning DAG: {dag.session_id}", show_header=True)
    table.add_column("Attribute", style="cyan")
    table.add_column("Value")

    table.add_row("Nodes", str(len(dag.nodes)))
    table.add_row("Edges", str(len(dag.edges)))
    table.add_row("Pivot nodes", str(len(dag.pivot_nodes)))
    table.add_row("Noise ratio", f"{dag.noise_ratio:.1%}")
    table.add_row("", "")
    for ntype, count in sorted(type_counts.items()):
        table.add_row(f"  {ntype}", str(count))

    console.print(table)

    if dag.pivot_nodes:
        console.print("\n[bold yellow]⚡ Pivot points:[/bold yellow]")
        for node in dag.nodes:
            if node.node_id in dag.pivot_nodes:
                console.print(f"  • [{node.node_id}] {node.summary}")

    solutions = [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]
    if solutions:
        console.print("\n[bold green]✓ Solutions reached:[/bold green]")
        for node in solutions:
            console.print(f"  • [{node.node_id}] {node.summary}")

    dead_ends = [n for n in dag.nodes if n.node_type == NodeType.DEAD_END]
    if dead_ends:
        console.print("\n[bold red]✗ Dead ends:[/bold red]")
        for node in dead_ends:
            console.print(f"  • [{node.node_id}] {node.summary}")


async def main():
    parser = argparse.ArgumentParser(description="Run the cognitive memory pipeline")
    parser.add_argument("session", help="Path to a Claude Code JSONL session file")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM extraction (parse only)")
    parser.add_argument("--save-dag", help="Save DAG to JSON file")
    parser.add_argument("--save-session", help="Save NormalizedSession to JSON file")
    args = parser.parse_args()

    session_path = Path(args.session)
    if not session_path.exists():
        console.print(f"[red]File not found: {session_path}[/red]")
        sys.exit(1)

    # Step 1: Parse
    console.rule("[bold]Step 1: Parsing session")
    cc_parser = ClaudeCodeParser()
    session = cc_parser.parse_file(session_path)
    print_session_summary(session)

    if args.save_session:
        Path(args.save_session).write_text(session.model_dump_json(indent=2))
        console.print(f"[green]Session saved to {args.save_session}[/green]")

    if args.no_llm:
        console.print("\n[yellow]Skipping LLM extraction (--no-llm)[/yellow]")
        return

    # Step 2: Extract DAG
    has_llm_credentials = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
    )
    if not has_llm_credentials:
        console.print("\n[yellow]No LLM credentials set -- skipping LLM extraction.[/yellow]")
        console.print("Set ANTHROPIC_API_KEY or AWS credentials to run full DAG extraction.")
        return

    console.rule("[bold]Step 2: Extracting reasoning DAG")
    console.print(f"Processing {len(session.messages)} messages in overlapping windows...")

    builder = DAGBuilder()
    dag = await builder.build(session)
    print_dag_summary(dag)

    if args.save_dag:
        Path(args.save_dag).write_text(dag.model_dump_json(indent=2))
        console.print(f"\n[green]DAG saved to {args.save_dag}[/green]")

    console.rule("[bold green]Pipeline complete")


if __name__ == "__main__":
    asyncio.run(main())
