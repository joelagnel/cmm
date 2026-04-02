#!/usr/bin/env python3
"""CLI query interface for cognitive memory — used by Claude Code slash commands.

This module provides the same functionality as the MCP server tools but as
direct CLI commands, so they can be invoked from Claude Code skill files
without requiring an MCP server.

Every invocation is automatically logged to data/eval/interactions.db
for evaluation metrics. Logging is fail-safe — if it breaks, retrieval
still works.

Usage:
    python -m src.delivery.cli_query profile
    python -m src.delivery.cli_query pitfalls
    python -m src.delivery.cli_query search "offline sync"
    python -m src.delivery.cli_query diagnose "test collection fails"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.store.vector_store import MemoryStore
from src.delivery.mcp_server import (
    _fmt_profile,
    _fmt_pitfalls,
    _fmt_search_results,
    _fmt_strategies,
)

_DEFAULT_STORE = str(Path(__file__).parent.parent.parent / "data" / "memory_store")


def _store(args) -> MemoryStore:
    return MemoryStore(persist_dir=args.store_dir)


def _logger():
    """Get the interaction logger. Returns None on failure."""
    try:
        from src.evaluation.logger import InteractionLogger
        return InteractionLogger()
    except Exception:
        return None


def _log(logger, skill, project_id, query_text=None, results=None, response_time_ms=0, output=""):
    """Log an invocation. Fails silently."""
    if logger is None:
        return
    try:
        logger.log_invocation(
            skill=skill,
            project_id=project_id,
            query_text=query_text,
            results=results,
            response_time_ms=response_time_ms,
            raw_output_len=len(output),
        )
    except Exception:
        pass


def cmd_profile(args):
    logger = _logger()
    t0 = time.perf_counter()

    store = _store(args)
    profile = store.get_profile(args.project)

    elapsed = (time.perf_counter() - t0) * 1000

    if profile is None:
        count = store.node_count(args.project)
        if count == 0:
            output = f"No cognitive profile found for `{args.project}`. Run the ingest pipeline first."
        else:
            output = f"No profile built yet for `{args.project}` ({count} nodes stored)."
        _log(logger, "cognitive-profile", args.project, response_time_ms=elapsed, output=output)
        print(output)
        return

    output = _fmt_profile(profile)
    _log(logger, "cognitive-profile", args.project, response_time_ms=elapsed, output=output)
    print(output)


def cmd_pitfalls(args):
    logger = _logger()
    t0 = time.perf_counter()

    store = _store(args)
    profile = store.get_profile(args.project)

    elapsed = (time.perf_counter() - t0) * 1000

    if profile is None:
        output = f"No profile found for `{args.project}`. Run the ingest pipeline first."
        _log(logger, "pitfalls", args.project, response_time_ms=elapsed, output=output)
        print(output)
        return

    output = _fmt_pitfalls(profile.pitfalls)
    _log(logger, "pitfalls", args.project, response_time_ms=elapsed, output=output)
    print(output)


def cmd_search(args):
    logger = _logger()
    t0 = time.perf_counter()

    store = _store(args)
    results = store.search(args.query, project_id=args.project, top_k=args.top_k)

    elapsed = (time.perf_counter() - t0) * 1000

    output = _fmt_search_results(results)
    _log(logger, "search-memory", args.project, query_text=args.query,
         results=results, response_time_ms=elapsed, output=output)
    print(output)


def cmd_diagnose(args):
    logger = _logger()
    t0 = time.perf_counter()

    store = _store(args)
    profile = store.get_profile(args.project)

    if profile is None:
        elapsed = (time.perf_counter() - t0) * 1000
        output = f"No profile found for `{args.project}`. Run the ingest pipeline first."
        _log(logger, "diagnose", args.project, query_text=args.problem,
             response_time_ms=elapsed, output=output)
        print(output)
        return

    if not profile.diagnostic_strategies:
        results = store.search(args.problem, project_id=args.project, top_k=3)
        elapsed = (time.perf_counter() - t0) * 1000
        output = "No structured diagnostic strategies yet. Relevant past reasoning:\n\n" + _fmt_search_results(results)
        _log(logger, "diagnose", args.project, query_text=args.problem,
             results=results, response_time_ms=elapsed, output=output)
        print(output)
        return

    desc_lower = args.problem.lower()
    ranked = sorted(
        profile.diagnostic_strategies,
        key=lambda s: sum(1 for w in desc_lower.split() if w in s.trigger.lower()),
        reverse=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    output = _fmt_strategies(ranked[:3], args.problem)
    _log(logger, "diagnose", args.project, query_text=args.problem,
         response_time_ms=elapsed, output=output)
    print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Query cognitive memory from the command line",
        prog="python -m src.delivery.cli_query",
    )
    parser.add_argument(
        "--project", "-p",
        default=os.environ.get("CMM_PROJECT_ID", ""),
        help="Project ID (default: CMM_PROJECT_ID env var)",
    )
    parser.add_argument(
        "--store-dir",
        default=os.environ.get("CMM_STORE_PATH", _DEFAULT_STORE),
        help="Memory store directory (default: CMM_STORE_PATH env var)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profile", help="Get full cognitive profile")
    sub.add_parser("pitfalls", help="Get known pitfalls ranked by severity")

    s = sub.add_parser("search", help="Search memory for relevant reasoning")
    s.add_argument("query", help="Search query")
    s.add_argument("--top-k", type=int, default=5, help="Number of results")

    d = sub.add_parser("diagnose", help="Find diagnostic strategies for a problem")
    d.add_argument("problem", help="Problem description")

    args = parser.parse_args()

    if not args.project:
        print("Error: provide --project or set CMM_PROJECT_ID environment variable.")
        sys.exit(1)

    {
        "profile": cmd_profile,
        "pitfalls": cmd_pitfalls,
        "search": cmd_search,
        "diagnose": cmd_diagnose,
    }[args.command](args)


if __name__ == "__main__":
    main()
