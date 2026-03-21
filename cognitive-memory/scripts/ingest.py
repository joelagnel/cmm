#!/usr/bin/env python3
"""Ingest one or more session files into the memory store and build a cognitive profile.

Usage:
    # Ingest a single session
    python scripts/ingest.py session.jsonl --project myproject

    # Ingest all sessions for a project
    python scripts/ingest.py fixtures/claude_code/*.jsonl --project mcp-gateway

    # Ingest + rebuild profile
    python scripts/ingest.py session.jsonl --project myproject --build-profile

    # Skip LLM extraction (embed & store with heuristic nodes only)
    python scripts/ingest.py session.jsonl --project myproject --no-llm

    # Show what's stored for a project
    python scripts/ingest.py --status --project myproject
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.ingestion import ClaudeCodeParser
from src.extraction import DAGBuilder
from src.compression import SemanticDeduplicator, ProfileBuilder
from src.store import MemoryStore
from src.schemas.reasoning import NodeType, ReasoningDAG, ReasoningNode

console = Console()

_DEFAULT_STORE = Path(__file__).parent.parent / "data" / "memory_store"


def _heuristic_dag(session) -> ReasoningDAG:
    """Fast heuristic DAG: one node per non-trivial assistant turn."""
    from src.schemas.reasoning import ReasoningEdge
    nodes = []
    for i, msg in enumerate(session.messages):
        from src.schemas.session import MessageRole
        if msg.role == MessageRole.ASSISTANT and len(msg.content) > 50:
            ntype = NodeType.CONTEXT_LOAD
            if any(kw in msg.content.lower() for kw in ["hypothesis", "suspect", "think", "believe"]):
                ntype = NodeType.HYPOTHESIS
            elif any(kw in msg.content.lower() for kw in ["found", "discovered", "notice", "unexpected"]):
                ntype = NodeType.DISCOVERY
            elif any(kw in msg.content.lower() for kw in ["pivot", "instead", "approach", "wrong"]):
                ntype = NodeType.PIVOT
            elif any(kw in msg.content.lower() for kw in ["fixed", "resolved", "solution", "works now"]):
                ntype = NodeType.SOLUTION
            elif any(kw in msg.content.lower() for kw in ["failed", "didn't work", "no luck", "not working"]):
                ntype = NodeType.DEAD_END

            nodes.append(ReasoningNode(
                node_id=f"hnode-{len(nodes):03d}",
                node_type=ntype,
                summary=msg.content[:200],
                evidence="heuristic extraction",
                message_range=(i, i + 1),
                confidence=0.4,
            ))

    edges = [
        ReasoningEdge(source_id=nodes[i].node_id, target_id=nodes[i+1].node_id, relationship="led_to")
        for i in range(len(nodes) - 1)
    ]
    pivots = [n.node_id for n in nodes if n.node_type == NodeType.PIVOT]
    original = len(session.messages)
    return ReasoningDAG(
        session_id=session.session_id,
        nodes=nodes,
        edges=edges,
        pivot_nodes=pivots,
        noise_ratio=1.0 - (len(nodes) / original) if original else 0,
    )


async def ingest_session(
    path: Path,
    project_id: str,
    store: MemoryStore,
    use_llm: bool,
    api_key: str | None,
) -> dict:
    cc_parser = ClaudeCodeParser()
    session = cc_parser.parse_file(path)

    if use_llm and api_key:
        builder = DAGBuilder(api_key=api_key)
        dag = await builder.build(session)
        method = "LLM"
    else:
        dag = _heuristic_dag(session)
        method = "heuristic"

    # Deduplicate before storing
    dedup = SemanticDeduplicator(store)
    result = dedup.deduplicate(dag.nodes, project_id, session.session_id)

    # Store only non-dropped nodes
    dag_to_store = ReasoningDAG(
        session_id=dag.session_id,
        nodes=result.stored,
        edges=dag.edges,
        pivot_nodes=dag.pivot_nodes,
        noise_ratio=dag.noise_ratio,
    )
    stored_count = store.store_dag(dag_to_store, project_id)

    return {
        "session_id": session.session_id,
        "messages": len(session.messages),
        "dag_nodes": len(dag.nodes),
        "stored": stored_count,
        "merged": len(result.merged),
        "dropped": len(result.dropped),
        "noise_ratio": dag.noise_ratio,
        "method": method,
    }


def print_status(store: MemoryStore, project_id: str):
    table = Table(title=f"Memory store: {project_id}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    count = store.node_count(project_id)
    table.add_row("Stored nodes", str(count))

    profile = store.get_profile(project_id)
    if profile:
        table.add_row("Profile sessions", str(profile.session_count))
        table.add_row("Architectural insights", str(len(profile.architectural_insights)))
        table.add_row("Pitfalls", str(len(profile.pitfalls)))
        table.add_row("Diagnostic strategies", str(len(profile.diagnostic_strategies)))
        table.add_row("Key patterns", str(len(profile.key_patterns)))
    else:
        table.add_row("Profile", "[yellow]not built yet[/yellow]")

    console.print(table)

    if profile and profile.pitfalls:
        console.print("\n[bold red]Known pitfalls:[/bold red]")
        for p in profile.pitfalls:
            console.print(f"  [{p.severity.upper()}] {p.description}")

    if profile and profile.architectural_insights:
        console.print("\n[bold blue]Architectural insights:[/bold blue]")
        for ins in profile.architectural_insights:
            console.print(f"  [{ins.component}] {ins.insight}")


async def main():
    parser = argparse.ArgumentParser(description="Ingest sessions into memory store")
    parser.add_argument("sessions", nargs="*", help="JSONL session files to ingest")
    parser.add_argument("--project", "-p", required=True, help="Project ID")
    parser.add_argument("--store-dir", default=str(_DEFAULT_STORE), help="Memory store directory")
    parser.add_argument("--build-profile", action="store_true", help="Build cognitive profile after ingestion")
    parser.add_argument("--no-llm", action="store_true", help="Use heuristic extraction (no API calls)")
    parser.add_argument("--status", action="store_true", help="Show current store status and exit")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_llm = not args.no_llm and bool(api_key)

    store = MemoryStore(persist_dir=args.store_dir)

    if args.status:
        print_status(store, args.project)
        return

    if not args.sessions:
        console.print("[red]No session files provided.[/red]")
        parser.print_help()
        sys.exit(1)

    # Ingest each session
    table = Table(title=f"Ingestion results — project: {args.project}", show_header=True)
    table.add_column("File", style="cyan", max_width=30)
    table.add_column("Messages")
    table.add_column("Nodes")
    table.add_column("Stored")
    table.add_column("Merged")
    table.add_column("Dropped")
    table.add_column("Noise%")
    table.add_column("Method")

    for session_path in args.sessions:
        path = Path(session_path)
        if not path.exists():
            console.print(f"[yellow]Skipping missing file: {path}[/yellow]")
            continue

        console.print(f"Ingesting [cyan]{path.name}[/cyan]...", end=" ")
        try:
            stats = await ingest_session(path, args.project, store, use_llm, api_key)
            console.print(f"[green]done[/green] ({stats['stored']} stored)")
            table.add_row(
                path.name,
                str(stats["messages"]),
                str(stats["dag_nodes"]),
                str(stats["stored"]),
                str(stats["merged"]),
                str(stats["dropped"]),
                f"{stats['noise_ratio']:.0%}",
                stats["method"],
            )
        except Exception as e:
            console.print(f"[red]failed: {e}[/red]")

    console.print()
    console.print(table)
    console.print(f"\nTotal nodes in store: [bold]{store.node_count(args.project)}[/bold]")

    if args.build_profile:
        if not api_key:
            console.print("\n[yellow]ANTHROPIC_API_KEY not set — cannot build profile.[/yellow]")
        else:
            console.rule("[bold]Building cognitive profile")
            pb = ProfileBuilder(api_key=api_key)
            profile = await pb.build_profile(args.project, store)
            console.print(f"[green]Profile built:[/green]")
            console.print(f"  {len(profile.architectural_insights)} insights")
            console.print(f"  {len(profile.pitfalls)} pitfalls")
            console.print(f"  {len(profile.diagnostic_strategies)} diagnostic strategies")
            console.print(f"  {len(profile.key_patterns)} key patterns")
            print_status(store, args.project)


if __name__ == "__main__":
    asyncio.run(main())
