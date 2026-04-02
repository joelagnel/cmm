#!/usr/bin/env python3
"""Evaluation dashboard — pulls together interaction logs and session evaluations.

Usage:
    python scripts/eval_report.py
    python scripts/eval_report.py --project supply-chain
    python scripts/eval_report.py --json output/eval_report.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.evaluation.logger import InteractionLogger

console = Console()

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "eval" / "interactions.db"


def build_report(logger: InteractionLogger, project_id: str | None = None) -> dict:
    """Build the full evaluation report."""
    stats = logger.get_stats(project_id)
    invocations = logger.get_invocations(project_id=project_id, limit=500)
    evaluations = logger.get_session_evaluations(project_id=project_id)

    # Skill breakdown
    skill_counts: dict[str, int] = {}
    for inv in invocations:
        skill = inv.get("skill", "unknown")
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

    # Session evaluation aggregates for memory vs no-memory
    memory_sessions = [e for e in evaluations if e.get("memory_used_at_start")]
    no_memory_sessions = [e for e in evaluations if not e.get("memory_used_at_start")]

    def _avg(items, key):
        vals = [e.get(key, 0) for e in items]
        return round(sum(vals) / len(vals), 1) if vals else 0

    comparison = {}
    if memory_sessions and no_memory_sessions:
        comparison = {
            "memory_sessions": len(memory_sessions),
            "no_memory_sessions": len(no_memory_sessions),
            "avg_dead_ends_with_memory": _avg(memory_sessions, "total_dead_ends"),
            "avg_dead_ends_without_memory": _avg(no_memory_sessions, "total_dead_ends"),
            "avg_pivots_with_memory": _avg(memory_sessions, "total_pivots"),
            "avg_pivots_without_memory": _avg(no_memory_sessions, "total_pivots"),
            "avg_msgs_to_solution_with_memory": _avg(memory_sessions, "messages_to_first_solution"),
            "avg_msgs_to_solution_without_memory": _avg(no_memory_sessions, "messages_to_first_solution"),
        }

    return {
        "project_id": project_id or "all",
        "stats": stats,
        "skill_breakdown": skill_counts,
        "evaluations_count": len(evaluations),
        "comparison": comparison,
    }


def print_report(report: dict):
    """Print the report using rich formatting."""
    project = report["project_id"]
    stats = report["stats"]

    console.print()
    console.print(Panel(
        f"[bold]Memory System Evaluation Report[/bold]\n"
        f"Project: [cyan]{project}[/cyan]",
        style="blue",
    ))

    # Retrieval metrics
    table = Table(title="Retrieval Metrics", show_header=False)
    table.add_column("Metric", style="cyan", min_width=30)
    table.add_column("Value", justify="right")

    table.add_row("Total interactions logged", str(stats.get("total_invocations", 0)))
    table.add_row("Total sessions evaluated", str(stats.get("total_sessions_evaluated", 0)))

    avg_sim = stats.get("avg_similarity")
    table.add_row("Average similarity score", f"{avg_sim:.3f}" if avg_sim else "N/A")

    avg_rt = stats.get("avg_response_time_ms")
    table.add_row("Average response time", f"{avg_rt:.0f}ms" if avg_rt else "N/A")

    console.print(table)

    # Skill breakdown
    if report["skill_breakdown"]:
        skill_table = Table(title="Skill Usage", show_header=True)
        skill_table.add_column("Skill", style="cyan")
        skill_table.add_column("Invocations", justify="right")
        for skill, count in sorted(report["skill_breakdown"].items(), key=lambda x: -x[1]):
            skill_table.add_row(skill, str(count))
        console.print(skill_table)

    # Helpfulness
    pitfalls_surfaced = stats.get("total_pitfalls_surfaced", 0)
    pitfalls_avoided = stats.get("total_pitfalls_avoided", 0)
    errors_resolved = stats.get("total_errors_resolved", 0)
    early_memory = stats.get("sessions_with_early_memory", 0)

    if pitfalls_surfaced or errors_resolved:
        help_table = Table(title="Helpfulness Signals", show_header=False)
        help_table.add_column("Metric", style="cyan", min_width=30)
        help_table.add_column("Value", justify="right")

        help_table.add_row("Sessions with early memory load", str(early_memory))
        help_table.add_row("Errors resolved via memory", str(errors_resolved))
        help_table.add_row("Pitfalls surfaced", str(pitfalls_surfaced))
        if pitfalls_surfaced > 0:
            rate = pitfalls_avoided / pitfalls_surfaced * 100
            help_table.add_row(
                "Pitfalls avoided",
                f"{pitfalls_avoided}/{pitfalls_surfaced} ({rate:.0f}%)",
            )
        console.print(help_table)

    # Memory vs no-memory comparison
    comp = report.get("comparison", {})
    if comp:
        comp_table = Table(title="Efficiency: Memory vs No-Memory Sessions", show_header=True)
        comp_table.add_column("Metric", style="cyan")
        comp_table.add_column("With Memory", justify="right", style="green")
        comp_table.add_column("Without Memory", justify="right", style="red")

        comp_table.add_row(
            "Avg dead ends",
            str(comp.get("avg_dead_ends_with_memory", "?")),
            str(comp.get("avg_dead_ends_without_memory", "?")),
        )
        comp_table.add_row(
            "Avg pivots",
            str(comp.get("avg_pivots_with_memory", "?")),
            str(comp.get("avg_pivots_without_memory", "?")),
        )
        comp_table.add_row(
            "Avg msgs to solution",
            str(comp.get("avg_msgs_to_solution_with_memory", "?")),
            str(comp.get("avg_msgs_to_solution_without_memory", "?")),
        )
        console.print(comp_table)

    console.print()


def main():
    parser = argparse.ArgumentParser(description="Evaluation report for cognitive memory")
    parser.add_argument("--project", "-p", default=None, help="Project ID filter")
    parser.add_argument("--db", default=str(_DEFAULT_DB), help="Path to interactions.db")
    parser.add_argument("--json", dest="json_output", default=None, help="Export as JSON file")
    args = parser.parse_args()

    logger = InteractionLogger(db_path=args.db)
    report = build_report(logger, project_id=args.project)

    print_report(report)

    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(report, indent=2))
        console.print(f"[dim]JSON exported to {args.json_output}[/dim]")


if __name__ == "__main__":
    main()
