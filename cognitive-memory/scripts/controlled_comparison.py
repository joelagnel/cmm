#!/usr/bin/env python3
"""Controlled A/B comparison: same task, same codebase, memory vs no-memory.

Unlike the correlational comparison in eval_report.py (which splits sessions
by "memory_used_at_start"), this script provides causal evidence by running
the same prompt twice under identical conditions — once with cognitive memory
injected, once without — and comparing the resulting DAGs.

Usage:
    python scripts/controlled_comparison.py \
        --project supply-chain \
        --prompt "Fix the failing test in tests/test_parser.py" \
        --store-dir data/memory_store

    # Or with a prompt file:
    python scripts/controlled_comparison.py \
        --project supply-chain \
        --prompt-file prompts/fix_parser.md

    # Dry run — just show what would be compared:
    python scripts/controlled_comparison.py --dry-run --project supply-chain

The script:
    1. Captures the current git state (branch + HEAD sha)
    2. Runs the prompt WITHOUT memory → captures session JSONL
    3. Resets to the same git state
    4. Runs the prompt WITH memory (injects cached_profile + search results)
    5. Extracts DAGs from both sessions
    6. Compares: dead ends, pivots, solutions, messages to solution, duration
    7. Outputs a structured comparison report (JSON + terminal)

NOTE: This script is a *framework* — it generates the two prompts and
comparison harness. The actual agent execution depends on the user's
Claude Code setup. It writes two prompt files and a comparison runner
that the user can execute manually or wire into CI.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def _generate_baseline_prompt(task_prompt: str) -> str:
    """Generate a prompt that explicitly disables memory."""
    return f"""\
IMPORTANT: Do NOT use any cognitive memory commands. Do NOT run /search-memory,
/cognitive-profile, /pitfalls, or /diagnose. Solve this task using only the
codebase as-is.

Task:
{task_prompt}
"""


def _generate_assisted_prompt(task_prompt: str, profile_content: str, search_results: str) -> str:
    """Generate a prompt with memory pre-injected."""
    return f"""\
Before starting, review this accumulated project knowledge from past sessions:

{profile_content}

Relevant past reasoning for this task:
{search_results}

You may also use /search-memory or /diagnose during the task if needed.

Task:
{task_prompt}
"""


def _compare_dags(baseline_dag, assisted_dag) -> dict:
    """Compare two DAGs and compute improvement metrics."""
    from src.schemas.reasoning import NodeType

    def _stats(dag):
        dead_ends = sum(1 for n in dag.nodes if n.node_type == NodeType.DEAD_END)
        pivots = sum(1 for n in dag.nodes if n.node_type == NodeType.PIVOT)
        solutions = sum(1 for n in dag.nodes if n.node_type == NodeType.SOLUTION)
        hypotheses = sum(1 for n in dag.nodes if n.node_type == NodeType.HYPOTHESIS)

        msgs_to_solution = 0
        solution_nodes = [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]
        if solution_nodes:
            msgs_to_solution = min(s.message_range[0] for s in solution_nodes)

        return {
            "total_nodes": len(dag.nodes),
            "dead_ends": dead_ends,
            "pivots": pivots,
            "solutions": solutions,
            "hypotheses": hypotheses,
            "messages_to_first_solution": msgs_to_solution,
        }

    def _reduction(before, after):
        if before == 0:
            return 0.0
        return round((before - after) / before * 100, 1)

    b = _stats(baseline_dag)
    a = _stats(assisted_dag)

    return {
        "baseline": b,
        "assisted": a,
        "reductions": {
            "dead_ends": _reduction(b["dead_ends"], a["dead_ends"]),
            "pivots": _reduction(b["pivots"], a["pivots"]),
            "messages_to_solution": _reduction(
                b["messages_to_first_solution"],
                a["messages_to_first_solution"],
            ),
            "total_nodes": _reduction(b["total_nodes"], a["total_nodes"]),
        },
        "verdict": (
            "memory_helped" if a["dead_ends"] < b["dead_ends"]
            else "no_difference" if a["dead_ends"] == b["dead_ends"]
            else "memory_hurt"
        ),
    }


def _print_comparison(comparison: dict, task: str):
    console.print(Panel(
        f"[bold]Controlled Comparison: Memory vs No-Memory[/bold]\n"
        f"Task: [cyan]{task[:80]}[/cyan]",
        style="blue",
    ))

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline (no memory)", justify="right")
    table.add_column("Assisted (with memory)", justify="right")
    table.add_column("Reduction", justify="right")

    b = comparison["baseline"]
    a = comparison["assisted"]
    r = comparison["reductions"]

    for metric, key in [
        ("Dead ends", "dead_ends"),
        ("Pivots", "pivots"),
        ("Msgs to solution", "messages_to_first_solution"),
        ("Total nodes", "total_nodes"),
    ]:
        bv = str(b[key])
        av = str(a[key])
        rv = r.get(key, 0)
        color = "green" if rv > 0 else ("red" if rv < 0 else "dim")
        table.add_row(metric, bv, av, f"[{color}]{rv:+.1f}%[/{color}]")

    console.print(table)

    verdict = comparison["verdict"]
    if verdict == "memory_helped":
        console.print("[bold green]Verdict: Memory helped (fewer dead ends)[/bold green]")
    elif verdict == "memory_hurt":
        console.print("[bold red]Verdict: Memory may have hurt (more dead ends)[/bold red]")
    else:
        console.print("[bold yellow]Verdict: No measurable difference[/bold yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Controlled A/B comparison of memory vs no-memory on the same task"
    )
    parser.add_argument("--project", "-p", required=True, help="Project ID")
    parser.add_argument("--prompt", default=None, help="Task prompt (inline)")
    parser.add_argument("--prompt-file", default=None, help="Task prompt (from file)")
    parser.add_argument("--store-dir", default=None, help="Memory store directory")
    parser.add_argument("--output", "-o", default="output/", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Generate prompt files only")
    parser.add_argument(
        "--baseline-session", default=None,
        help="Path to a pre-recorded baseline session JSONL (skip live baseline)"
    )
    parser.add_argument(
        "--assisted-session", default=None,
        help="Path to a pre-recorded assisted session JSONL (skip live assisted)"
    )
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        console.print("[red]Provide --prompt or --prompt-file[/red]")
        sys.exit(1)

    task_prompt = args.prompt
    if args.prompt_file:
        task_prompt = Path(args.prompt_file).read_text().strip()

    store_dir = args.store_dir or str(Path(__file__).parent.parent / "data" / "memory_store")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate the two prompts
    baseline_prompt = _generate_baseline_prompt(task_prompt)

    # Load memory for the assisted prompt
    from src.store.vector_store import MemoryStore
    from src.delivery.mcp_server import _fmt_profile, _fmt_search_results

    store = MemoryStore(persist_dir=store_dir)
    profile = store.get_profile(args.project)
    profile_text = _fmt_profile(profile) if profile else "(no profile available)"
    results = store.search(task_prompt, project_id=args.project, top_k=5)
    search_text = _fmt_search_results(results) if results else "(no search results)"

    assisted_prompt = _generate_assisted_prompt(task_prompt, profile_text, search_text)

    # Write prompts
    (out_dir / "baseline_prompt.md").write_text(baseline_prompt)
    (out_dir / "assisted_prompt.md").write_text(assisted_prompt)
    console.print(f"[green]Wrote prompts to {out_dir}/[/green]")

    if args.dry_run:
        console.print()
        console.print("[yellow]DRY RUN — run these prompts manually, then re-run with:[/yellow]")
        console.print(f"  python scripts/controlled_comparison.py \\")
        console.print(f"    --project {args.project} \\")
        console.print(f"    --prompt '{task_prompt[:60]}...' \\")
        console.print(f"    --baseline-session output/baseline.jsonl \\")
        console.print(f"    --assisted-session output/assisted.jsonl")
        return

    # If pre-recorded sessions provided, compare them
    if args.baseline_session and args.assisted_session:
        from src.ingestion import ClaudeCodeParser
        from src.extraction.warm_extractor import WarmExtractor

        parser = ClaudeCodeParser()
        extractor = WarmExtractor()

        baseline_session = parser.parse_file(Path(args.baseline_session))
        assisted_session = parser.parse_file(Path(args.assisted_session))

        baseline_dag = extractor.extract(baseline_session)
        assisted_dag = extractor.extract(assisted_session)

        comparison = _compare_dags(baseline_dag, assisted_dag)
        comparison["task"] = task_prompt
        comparison["project"] = args.project
        comparison["timestamp"] = datetime.now(timezone.utc).isoformat()

        _print_comparison(comparison, task_prompt)

        report_path = out_dir / f"comparison_{args.project}_{int(time.time())}.json"
        report_path.write_text(json.dumps(comparison, indent=2))
        console.print(f"\n[dim]Report: {report_path}[/dim]")
    else:
        console.print()
        console.print("[yellow]No session files provided. Run the prompts manually:[/yellow]")
        console.print(f"  1. Start Claude Code and paste {out_dir}/baseline_prompt.md")
        console.print(f"  2. Save the session JSONL as {out_dir}/baseline.jsonl")
        console.print(f"  3. Reset git state (git stash or git checkout)")
        console.print(f"  4. Start Claude Code and paste {out_dir}/assisted_prompt.md")
        console.print(f"  5. Save the session JSONL as {out_dir}/assisted.jsonl")
        console.print(f"  6. Re-run this script with --baseline-session and --assisted-session")


if __name__ == "__main__":
    main()
