#!/usr/bin/env python3
"""Visualize a ReasoningDAG as a Mermaid diagram or Graphviz dot file.

Usage:
    python scripts/visualize_dag.py <session.jsonl> [--format mermaid|dot]
    python scripts/visualize_dag.py --dag-json <dag.json> [--format mermaid|dot]
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import ClaudeCodeParser
from src.extraction import DAGBuilder
from src.schemas.reasoning import NodeType, ReasoningDAG

# Node type → color mapping for visualization
_COLORS = {
    NodeType.HYPOTHESIS: "#AED6F1",       # light blue
    NodeType.INVESTIGATION: "#A9DFBF",    # light green
    NodeType.DISCOVERY: "#F9E79F",        # yellow
    NodeType.PIVOT: "#F1948A",            # salmon/red
    NodeType.SOLUTION: "#82E0AA",         # bright green
    NodeType.DEAD_END: "#D7BDE2",         # light purple
    NodeType.CONTEXT_LOAD: "#D5D8DC",    # grey
}

_SHAPES = {
    NodeType.HYPOTHESIS: "([{}])",
    NodeType.INVESTIGATION: "[{}]",
    NodeType.DISCOVERY: "{{{}}}",
    NodeType.PIVOT: "[/{}\\]",
    NodeType.SOLUTION: "([{}])",
    NodeType.DEAD_END: "[x{}x]",
    NodeType.CONTEXT_LOAD: "({})",
}


def render_mermaid(dag: ReasoningDAG) -> str:
    """Render the DAG as a Mermaid flowchart."""
    lines = ["flowchart TD"]

    for node in dag.nodes:
        node_id = node.node_id.replace("-", "_")
        label = node.summary[:60].replace('"', "'")
        if len(node.summary) > 60:
            label += "..."
        type_label = f"[{node.node_type.value.upper()}]\\n{label}"

        # Bold pivot nodes
        if node.node_id in dag.pivot_nodes:
            type_label = f"⚡{type_label}"

        lines.append(f'    {node_id}["{type_label}"]')

    for edge in dag.edges:
        src = edge.source_id.replace("-", "_")
        tgt = edge.target_id.replace("-", "_")
        rel = edge.relationship
        lines.append(f"    {src} -->|{rel}| {tgt}")

    # Style definitions
    lines.append("")
    for node in dag.nodes:
        node_id = node.node_id.replace("-", "_")
        color = _COLORS.get(node.node_type, "#FFFFFF")
        lines.append(f"    style {node_id} fill:{color}")

    return "\n".join(lines)


def render_dot(dag: ReasoningDAG) -> str:
    """Render the DAG as a Graphviz dot file."""
    lines = ["digraph reasoning_dag {"]
    lines.append("    rankdir=TB;")
    lines.append('    node [shape=box, style=filled, fontname="Helvetica"];')
    lines.append("")

    for node in dag.nodes:
        label = node.summary[:60].replace('"', '\\"')
        if len(node.summary) > 60:
            label += "..."
        type_label = f"{node.node_type.value.upper()}\\n{label}"
        color = _COLORS.get(node.node_type, "#FFFFFF")
        penwidth = "3" if node.node_id in dag.pivot_nodes else "1"
        lines.append(
            f'    "{node.node_id}" [label="{type_label}", fillcolor="{color}", penwidth={penwidth}];'
        )

    lines.append("")
    for edge in dag.edges:
        lines.append(
            f'    "{edge.source_id}" -> "{edge.target_id}" [label="{edge.relationship}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def print_summary(dag: ReasoningDAG):
    """Print a text summary of the DAG to stderr."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Session: {dag.session_id}", file=sys.stderr)
    print(f"Nodes: {len(dag.nodes)}  |  Edges: {len(dag.edges)}", file=sys.stderr)
    print(f"Pivots: {len(dag.pivot_nodes)}  |  Noise filtered: {dag.noise_ratio:.0%}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    type_counts: dict[str, int] = {}
    for node in dag.nodes:
        type_counts[node.node_type.value] = type_counts.get(node.node_type.value, 0) + 1

    for ntype, count in sorted(type_counts.items()):
        marker = "⚡" if ntype in ("pivot", "dead_end") else "  "
        print(f"  {marker}{ntype:20s} {count}", file=sys.stderr)

    print(f"\nPivot nodes:", file=sys.stderr)
    for node in dag.nodes:
        if node.node_id in dag.pivot_nodes:
            print(f"  • [{node.node_id}] {node.summary[:80]}", file=sys.stderr)

    print(f"\nSolution nodes:", file=sys.stderr)
    for node in dag.nodes:
        if node.node_type == NodeType.SOLUTION:
            print(f"  ✓ [{node.node_id}] {node.summary[:80]}", file=sys.stderr)
    print("", file=sys.stderr)


async def main():
    parser = argparse.ArgumentParser(description="Visualize a ReasoningDAG")
    parser.add_argument("session", nargs="?", help="Path to a Claude Code JSONL session file")
    parser.add_argument("--dag-json", help="Path to a pre-computed DAG JSON file")
    parser.add_argument("--format", choices=["mermaid", "dot"], default="mermaid")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.dag_json:
        dag = ReasoningDAG.model_validate_json(Path(args.dag_json).read_text())
    elif args.session:
        session_path = Path(args.session)
        print(f"Parsing session: {session_path}", file=sys.stderr)
        cc_parser = ClaudeCodeParser()
        session = cc_parser.parse_file(session_path)
        print(f"Messages: {len(session.messages)}", file=sys.stderr)

        print("Extracting reasoning DAG (this makes LLM calls)...", file=sys.stderr)
        builder = DAGBuilder(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        dag = await builder.build(session)
    else:
        parser.print_help()
        sys.exit(1)

    print_summary(dag)

    if args.format == "mermaid":
        output = render_mermaid(dag)
    else:
        output = render_dot(dag)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    asyncio.run(main())
