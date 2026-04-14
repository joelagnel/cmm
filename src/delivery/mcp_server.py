"""MCP server exposing cognitive memory to any compatible AI coding agent."""
from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

from mcp.server.fastmcp import FastMCP

from ..store.vector_store import MemoryStore

# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="cognitive-memory",
    instructions=dedent("""\
        This server provides persistent cognitive memory built from past coding sessions.

        Use it to:
        - `get_cognitive_profile` at the START of a session to load accumulated project knowledge.
        - `search_memory` when you hit a problem to find relevant past reasoning.
        - `get_pitfalls` before making significant changes to avoid known traps.
        - `get_diagnostic_strategy` when debugging to find proven approaches.
    """),
)

# Lazy-initialized store — created on first tool call using env vars
_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        store_path = os.environ.get(
            "CMM_STORE_PATH",
            str(Path.home() / ".cognitive-memory" / "store"),
        )
        _store = MemoryStore(persist_dir=store_path)
    return _store


def _default_project() -> str | None:
    return os.environ.get("CMM_PROJECT_ID")


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_search_results(results: list[dict]) -> str:
    if not results:
        return "No relevant memories found."
    lines = ["## Relevant past reasoning\n"]
    for i, r in enumerate(results, 1):
        sim = r.get("similarity", 0)
        ntype = r.get("node_type", "unknown")
        session = r.get("session_id", "?")
        pivot_marker = " ⚡" if r.get("is_pivot") else ""
        lines.append(
            f"**{i}. [{ntype.upper()}{pivot_marker}]** (similarity: {sim:.2f}, session: `{session}`)\n"
            f"{r['summary']}\n"
        )
    return "\n".join(lines)


def _fmt_profile(profile) -> str:
    lines = [f"# Cognitive Profile: `{profile.project_id}`\n"]
    lines.append(f"*Built from {profile.session_count} sessions · Updated {profile.last_updated.strftime('%Y-%m-%d')}*\n")

    if profile.architectural_insights:
        lines.append("## Architectural Insights\n")
        for ins in profile.architectural_insights:
            conf = f" (confidence: {ins.confidence:.0%})" if ins.confidence else ""
            lines.append(f"- **[{ins.component}]** {ins.insight}{conf}")
        lines.append("")

    if profile.pitfalls:
        lines.append("## Known Pitfalls\n")
        for p in profile.pitfalls:
            sev = p.severity.upper()
            lines.append(f"- **[{sev}]** {p.description}")
            if p.resolution_strategy:
                lines.append(f"  - Resolution: {p.resolution_strategy}")
        lines.append("")

    if profile.diagnostic_strategies:
        lines.append("## Diagnostic Strategies\n")
        for s in profile.diagnostic_strategies:
            sr = f" (success rate: {s.success_rate:.0%})" if s.success_rate else ""
            lines.append(f"- **Trigger:** {s.trigger}{sr}")
            for step in s.steps:
                lines.append(f"  1. {step}")
        lines.append("")

    if profile.key_patterns:
        lines.append("## Key Patterns\n")
        for p in profile.key_patterns:
            lines.append(f"- {p}")
        lines.append("")

    if profile.anti_patterns:
        lines.append("## Anti-Patterns\n")
        for p in profile.anti_patterns:
            lines.append(f"- {p}")

    return "\n".join(lines)


def _fmt_pitfalls(pitfalls) -> str:
    if not pitfalls:
        return "No known pitfalls recorded for this project."
    lines = ["## Known Pitfalls (ranked by severity)\n"]
    for p in pitfalls:
        sev = p.severity.upper()
        freq = f" · seen {p.frequency}x" if p.frequency > 1 else ""
        lines.append(f"**[{sev}]{freq}** {p.description}")
        if p.resolution_strategy:
            lines.append(f"  → {p.resolution_strategy}")
        lines.append("")
    return "\n".join(lines)


def _fmt_strategies(strategies: list, query: str) -> str:
    if not strategies:
        return f"No diagnostic strategies found for: {query}"
    lines = [f"## Diagnostic strategies for: {query}\n"]
    for s in strategies:
        sr = f" (success rate: {s.success_rate:.0%})" if s.success_rate else ""
        lines.append(f"**Trigger:** {s.trigger}{sr}")
        lines.append("**Steps:**")
        for i, step in enumerate(s.steps, 1):
            lines.append(f"  {i}. {step}")
        if s.source_sessions:
            lines.append(f"  *Derived from {len(s.source_sessions)} session(s)*")
        lines.append("")
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_memory(query: str, project_id: str = "", top_k: int = 5) -> str:
    """Search cognitive memory for reasoning patterns relevant to a problem.

    Returns reasoning fragments from past coding sessions semantically related
    to the query. Use this when you hit an unfamiliar error, need to understand
    a subsystem, or want to know if a similar problem has been solved before.

    Args:
        query: What you are looking for (error message, concept, component name)
        project_id: Project to search within. Defaults to CMM_PROJECT_ID env var.
        top_k: Number of results to return (default 5)
    """
    store = _get_store()
    pid = project_id or _default_project()
    results = store.search(query, project_id=pid, top_k=top_k)
    return _fmt_search_results(results)


@mcp.tool()
def get_cognitive_profile(project_id: str = "") -> str:
    """Get the full cognitive profile for a project.

    Returns accumulated knowledge: architectural insights, known pitfalls, and
    proven diagnostic strategies distilled from past sessions. Call this at the
    start of a session to orient yourself to the codebase.

    Args:
        project_id: Project to retrieve. Defaults to CMM_PROJECT_ID env var.
    """
    store = _get_store()
    pid = project_id or _default_project()
    if not pid:
        return "Error: provide project_id or set CMM_PROJECT_ID environment variable."

    profile = store.get_profile(pid)
    if profile is None:
        node_count = store.node_count(pid)
        if node_count == 0:
            return f"No cognitive profile found for `{pid}`. Run the ingest pipeline first."
        return (
            f"No profile built yet for `{pid}` ({node_count} nodes stored). "
            f"Run `python scripts/ingest.py --project {pid} --build-profile` to build one."
        )
    return _fmt_profile(profile)


@mcp.tool()
def get_pitfalls(project_id: str = "") -> str:
    """Get known pitfalls for a project, ranked by severity.

    Returns a ranked list of problems that previous agents encountered on this
    codebase, with resolution strategies. Check this before making significant
    changes to avoid repeating past mistakes.

    Args:
        project_id: Project to retrieve. Defaults to CMM_PROJECT_ID env var.
    """
    store = _get_store()
    pid = project_id or _default_project()
    if not pid:
        return "Error: provide project_id or set CMM_PROJECT_ID environment variable."

    profile = store.get_profile(pid)
    if profile is None:
        return f"No profile found for `{pid}`. Run the ingest pipeline first."

    return _fmt_pitfalls(profile.pitfalls)


@mcp.tool()
def get_diagnostic_strategy(problem_description: str, project_id: str = "") -> str:
    """Find proven diagnostic strategies for a problem type.

    Given a description of the problem you are facing, returns debugging
    approaches that worked in past sessions on this project.

    Args:
        problem_description: Describe the problem or error you are investigating
        project_id: Project to search. Defaults to CMM_PROJECT_ID env var.
    """
    store = _get_store()
    pid = project_id or _default_project()
    if not pid:
        return "Error: provide project_id or set CMM_PROJECT_ID environment variable."

    profile = store.get_profile(pid)
    if profile is None:
        return f"No profile found for `{pid}`. Run the ingest pipeline first."

    if not profile.diagnostic_strategies:
        # Fall back to semantic search over raw nodes
        results = store.search(problem_description, project_id=pid, top_k=3)
        return (
            "No structured diagnostic strategies yet. Relevant past reasoning:\n\n"
            + _fmt_search_results(results)
        )

    # Rank strategies by relevance to the problem description using simple keyword overlap
    desc_lower = problem_description.lower()
    ranked = sorted(
        profile.diagnostic_strategies,
        key=lambda s: sum(1 for w in desc_lower.split() if w in s.trigger.lower()),
        reverse=True,
    )
    return _fmt_strategies(ranked[:3], problem_description)


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    """Start the MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
