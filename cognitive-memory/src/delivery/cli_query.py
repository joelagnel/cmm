#!/usr/bin/env python3
"""CLI query interface for cognitive memory — used by Claude Code slash commands.

This module provides the same functionality as the MCP server tools but as
direct CLI commands, so they can be invoked from Claude Code skill files
without requiring an MCP server.

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


def cmd_profile(args):
    store = _store(args)
    profile = store.get_profile(args.project)
    if profile is None:
        count = store.node_count(args.project)
        if count == 0:
            print(f"No cognitive profile found for `{args.project}`. Run the ingest pipeline first.")
        else:
            print(f"No profile built yet for `{args.project}` ({count} nodes stored).")
        return
    print(_fmt_profile(profile))


def cmd_pitfalls(args):
    store = _store(args)
    profile = store.get_profile(args.project)
    if profile is None:
        print(f"No profile found for `{args.project}`. Run the ingest pipeline first.")
        return
    print(_fmt_pitfalls(profile.pitfalls))


def cmd_search(args):
    store = _store(args)
    results = store.search(args.query, project_id=args.project, top_k=args.top_k)
    print(_fmt_search_results(results))


def cmd_diagnose(args):
    store = _store(args)
    profile = store.get_profile(args.project)
    if profile is None:
        print(f"No profile found for `{args.project}`. Run the ingest pipeline first.")
        return

    if not profile.diagnostic_strategies:
        results = store.search(args.problem, project_id=args.project, top_k=3)
        print("No structured diagnostic strategies yet. Relevant past reasoning:\n")
        print(_fmt_search_results(results))
        return

    desc_lower = args.problem.lower()
    ranked = sorted(
        profile.diagnostic_strategies,
        key=lambda s: sum(1 for w in desc_lower.split() if w in s.trigger.lower()),
        reverse=True,
    )
    print(_fmt_strategies(ranked[:3], args.problem))


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
