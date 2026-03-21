"""Export a project's reasoning DAG (nodes + edges) as a Markdown report.

Usage
-----
# From ChromaDB store (nodes only — edges not persisted):
python scripts/export_dag_md.py -p supply-chain

# From a saved DAG JSON (includes edges):
python scripts/export_dag_md.py -p supply-chain --dag-json path/to/dag.json

# Custom store path or output file:
python scripts/export_dag_md.py -p supply-chain \
    --store-dir data/memory_store \
    --output reports/supply_chain_dag.md
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

# Make sure the project root is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Node type metadata ──────────────────────────────────────────────────────

_NODE_EMOJI = {
    "hypothesis":    "🔵",
    "investigation": "🔍",
    "discovery":     "✨",
    "pivot":         "🔄",
    "solution":      "✅",
    "dead_end":      "❌",
    "context_load":  "📖",
}

_NODE_DESC = {
    "hypothesis":    "Hypothesis",
    "investigation": "Investigation",
    "discovery":     "Discovery",
    "pivot":         "Pivot",
    "solution":      "Solution",
    "dead_end":      "Dead End",
    "context_load":  "Context Load",
}

_MERMAID_STYLE = {
    "hypothesis":    "fill:#60a5fa,stroke:#3b82f6,color:#fff",
    "investigation": "fill:#a78bfa,stroke:#7c3aed,color:#fff",
    "discovery":     "fill:#34d399,stroke:#059669,color:#fff",
    "pivot":         "fill:#fb923c,stroke:#ea580c,color:#fff",
    "solution":      "fill:#4ade80,stroke:#16a34a,color:#fff",
    "dead_end":      "fill:#f87171,stroke:#dc2626,color:#fff",
    "context_load":  "fill:#e5e7eb,stroke:#6b7280,color:#111",
}


# ── Markdown builders ───────────────────────────────────────────────────────

def _legend() -> str:
    rows = [f"| {_NODE_EMOJI[t]} | **{_NODE_DESC[t]}** |" for t in _NODE_EMOJI]
    return (
        "## Legend\n\n"
        "| Icon | Node Type |\n"
        "|------|-----------|\n"
        + "\n".join(rows)
        + "\n"
    )


def _summary_table(nodes: list[dict]) -> str:
    type_counts: dict[str, int] = {}
    pivot_count = 0
    for n in nodes:
        nt = n.get("node_type", "unknown")
        type_counts[nt] = type_counts.get(nt, 0) + 1
        if n.get("is_pivot"):
            pivot_count += 1

    rows = [
        f"| {_NODE_EMOJI.get(nt, '❓')} {_NODE_DESC.get(nt, nt)} | {cnt} |"
        for nt, cnt in sorted(type_counts.items(), key=lambda x: -x[1])
    ]
    return (
        "## Summary\n\n"
        f"- **Total nodes:** {len(nodes)}\n"
        f"- **Pivot nodes:** {pivot_count}\n"
        f"- **Sessions:** {len({n.get('session_id') for n in nodes})}\n\n"
        "| Node Type | Count |\n"
        "|-----------|-------|\n"
        + "\n".join(rows)
        + "\n"
    )


def _pivot_section(nodes: list[dict]) -> str:
    pivots = [n for n in nodes if n.get("is_pivot")]
    if not pivots:
        return ""
    lines = ["## ⚡ Pivot Points\n",
             "These are the key moments where the agent changed direction.\n"]
    for i, n in enumerate(pivots, 1):
        sid = n.get("session_id", "unknown")
        lines.append(f"### Pivot {i} — `{n.get('node_id', '?')}`")
        lines.append(f"*Session: {sid}*\n")
        lines.append(f"> {n.get('summary', '')}\n")
        ev = n.get("evidence", "")
        if ev:
            lines.append(f"**Evidence:** {ev}\n")
        lines.append("")
    return "\n".join(lines)


def _nodes_section(nodes: list[dict]) -> str:
    # Group by session
    sessions: dict[str, list[dict]] = {}
    for n in nodes:
        sid = n.get("session_id", "unknown")
        sessions.setdefault(sid, []).append(n)

    lines = ["## Nodes\n"]
    for sid, snodes in sessions.items():
        snodes_sorted = sorted(snodes, key=lambda x: x.get("msg_start", 0))
        lines.append(f"### Session: `{sid}`\n")
        lines.append("| # | Type | Node ID | Summary | Conf |")
        lines.append("|---|------|---------|---------|------|")
        for i, n in enumerate(snodes_sorted, 1):
            nt = n.get("node_type", "unknown")
            emoji = _NODE_EMOJI.get(nt, "❓")
            label = _NODE_DESC.get(nt, nt)
            nid = n.get("node_id", "?")
            pivot_marker = " ⚡" if n.get("is_pivot") else ""
            summary = n.get("summary", "").replace("|", "\\|")
            conf = f"{n.get('confidence', 0):.2f}"
            lines.append(f"| {i} | {emoji} {label}{pivot_marker} | `{nid}` | {summary} | {conf} |")
        lines.append("")
    return "\n".join(lines)


def _edges_section(edges: list[dict]) -> str:
    if not edges:
        return (
            "## Edges\n\n"
            "_Edges are not persisted to the vector store. "
            "Pass `--dag-json` with a saved DAG file to include edges._\n"
        )
    lines = [
        "## Edges\n",
        "| Source | Relationship | Target |",
        "|--------|:------------:|--------|",
    ]
    for e in edges:
        src = e.get("source_id", "?")
        tgt = e.get("target_id", "?")
        rel = e.get("relationship", "→")
        lines.append(f"| `{src}` | {rel} | `{tgt}` |")
    lines.append("")
    return "\n".join(lines)


def _mermaid_section(nodes: list[dict], edges: list[dict]) -> str:
    lines = ["## Mermaid Diagram\n", "```mermaid", "flowchart TD"]

    for n in nodes:
        nid = n.get("node_id", "unknown").replace("-", "_").replace(".", "_")
        nt = n.get("node_type", "context_load")
        # Truncate summary for the diagram
        raw = n.get("summary", "")
        label = raw[:60] + "…" if len(raw) > 60 else raw
        label = label.replace('"', "'")
        pivot_mark = " ⚡" if n.get("is_pivot") else ""
        lines.append(f'    {nid}["{_NODE_EMOJI.get(nt, "")} {label}{pivot_mark}"]')

    if edges:
        lines.append("")
        for e in edges:
            src = e.get("source_id", "").replace("-", "_").replace(".", "_")
            tgt = e.get("target_id", "").replace("-", "_").replace(".", "_")
            rel = e.get("relationship", "")
            if rel:
                lines.append(f"    {src} -->|{rel}| {tgt}")
            else:
                lines.append(f"    {src} --> {tgt}")
    else:
        # No edges: lay out nodes in sequence order within each session
        sessions: dict[str, list[dict]] = {}
        for n in nodes:
            sid = n.get("session_id", "unknown")
            sessions.setdefault(sid, []).append(n)
        lines.append("")
        for snodes in sessions.values():
            snodes_sorted = sorted(snodes, key=lambda x: x.get("msg_start", 0))
            for a, b in zip(snodes_sorted, snodes_sorted[1:]):
                aid = a.get("node_id", "").replace("-", "_").replace(".", "_")
                bid = b.get("node_id", "").replace("-", "_").replace(".", "_")
                lines.append(f"    {aid} --> {bid}")

    # Style classes
    lines.append("")
    lines.append("    classDef hypothesis    fill:#60a5fa,stroke:#3b82f6,color:#fff")
    lines.append("    classDef investigation fill:#a78bfa,stroke:#7c3aed,color:#fff")
    lines.append("    classDef discovery     fill:#34d399,stroke:#059669,color:#fff")
    lines.append("    classDef pivot         fill:#fb923c,stroke:#ea580c,color:#fff")
    lines.append("    classDef solution      fill:#4ade80,stroke:#16a34a,color:#fff")
    lines.append("    classDef dead_end      fill:#f87171,stroke:#dc2626,color:#fff")
    lines.append("    classDef context_load  fill:#e5e7eb,stroke:#6b7280,color:#111")
    lines.append("")

    for n in nodes:
        nid = n.get("node_id", "unknown").replace("-", "_").replace(".", "_")
        nt = n.get("node_type", "context_load")
        lines.append(f"    class {nid} {nt}")

    lines.append("```")
    return "\n".join(lines)


def _build_report(project_id: str, nodes: list[dict], edges: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"# Reasoning DAG — `{project_id}`\n",
        f"*Generated: {now}*\n",
        "---\n",
        _legend(),
        "---\n",
        _summary_table(nodes),
        "---\n",
        _pivot_section(nodes),
        "---\n" if any(n.get("is_pivot") for n in nodes) else "",
        _nodes_section(nodes),
        "---\n",
        _edges_section(edges),
        "---\n",
        _mermaid_section(nodes, edges),
    ]
    return "\n".join(s for s in sections if s)


# ── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--project", "-p", required=True, help="Project ID")
@click.option(
    "--store-dir",
    default=None,
    envvar="CMM_STORE_PATH",
    help="Memory store directory (default: data/memory_store)",
)
@click.option(
    "--dag-json",
    default=None,
    type=click.Path(exists=True),
    help="Path to a saved DAG JSON file (includes edges).",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output .md file path (default: reports/<project>_dag.md)",
)
@click.option("--stdout", is_flag=True, help="Print to stdout instead of a file.")
def main(project, store_dir, dag_json, output, stdout):
    """Export a project's reasoning DAG as a Markdown report."""
    from src.store.vector_store import MemoryStore

    # Resolve store path
    if store_dir is None:
        store_dir = Path(__file__).parent.parent / "data" / "memory_store"
    store = MemoryStore(persist_dir=store_dir)

    # Load nodes from store
    nodes = store.get_all_nodes(project)
    if not nodes:
        click.echo(f"[error] No nodes found for project '{project}' in {store_dir}", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(nodes)} nodes for project '{project}'.", err=True)

    # Load edges (only available if a DAG JSON was saved)
    edges: list[dict] = []
    if dag_json:
        with open(dag_json) as f:
            raw = json.load(f)
        edges = raw.get("edges", [])
        click.echo(f"Loaded {len(edges)} edges from {dag_json}.", err=True)

    # Build the report
    report = _build_report(project, nodes, edges)

    if stdout:
        print(report)
        return

    # Determine output path
    if output is None:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output = reports_dir / f"{project}_dag.md"
    else:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(report, encoding="utf-8")
    click.echo(f"Report written to: {output}", err=True)


if __name__ == "__main__":
    main()
