"""Claude Code hooks for automatic retrieval and ingestion.

Session-start hook:
    Fires when a new Claude Code session begins. Checks for .cmm/ folder,
    loads cached profile and runs a semantic search against ChromaDB using the
    project's llms.txt content. Outputs context for the agent.

Session-stop hook:
    Fires when a session ends. Finds the just-completed transcript, runs
    warm extraction, stores new nodes, and updates the manifest.

These are designed to be invoked as shell commands by Claude Code's hook system:
    "hooks": {
        "PreToolUse": [...],
        "PostToolUse": [...],
        "Stop": [{"command": "cmm hook stop"}]
    }

Or called programmatically from the CLI.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .project import CognitiveProject, discover_project, _COGNITIVE_DIR


def _get_store_path(project: CognitiveProject) -> str:
    """Resolve the ChromaDB store path from config or default."""
    configured = project.config.get("store_path")
    if configured:
        return configured

    # Default: ~/.cognitive-memory/store (shared across all projects)
    return str(Path.home() / ".cognitive-memory" / "store")


def _find_latest_session(project: CognitiveProject) -> Path | None:
    """Find the most recently modified JSONL session file for this project.

    Claude Code stores sessions at:
        ~/.claude/projects/<encoded-path>/<session-id>.jsonl
    """
    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.exists():
        return None

    # The encoded path replaces / with - and prepends -
    # e.g., /Users/me/Downloads/myproject → -Users-me-Downloads-myproject
    repo_path = project.repo_path.replace("/", "-")
    if not repo_path.startswith("-"):
        repo_path = "-" + repo_path

    # Look for matching project directory
    best_file = None
    best_mtime = 0.0

    for project_dir in claude_projects.iterdir():
        if not project_dir.is_dir():
            continue
        # Check if this directory corresponds to our project
        if project_dir.name == repo_path:
            for jsonl_file in project_dir.glob("*.jsonl"):
                mtime = jsonl_file.stat().st_mtime
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_file = jsonl_file

    return best_file


def session_start_hook(project_dir: Path | None = None) -> str:
    """Session-start hook: load context from .cmm/ folder.

    Returns a string to be injected into the agent's context.
    Returns empty string if no cognitive memory found.
    """
    project = discover_project(project_dir)
    if project is None:
        return (
            "No cognitive memory found for this project. "
            "Run 'cmm init' in the project root to enable persistent memory."
        )

    sections = []

    # 1. Load cached profile
    if project.cached_profile_path.exists():
        profile_content = project.cached_profile_path.read_text().strip()
        if profile_content and "No sessions ingested" not in profile_content:
            sections.append(profile_content)

    # 2. Semantic search using llms.txt content
    if project.config.get("auto_retrieve", True):
        store_path = _get_store_path(project)
        try:
            from src.store.vector_store import MemoryStore
            store = MemoryStore(persist_dir=store_path)
            node_count = store.node_count(project.project_id)

            if node_count > 0:
                # Search using the project description + llms.txt as query
                query_parts = [project.description]
                if project.llms_txt_path.exists():
                    llms_content = project.llms_txt_path.read_text()
                    # Extract key sections for search
                    for line in llms_content.split("\n"):
                        if line.startswith("- [") or line.startswith("- Trigger:"):
                            query_parts.append(line.strip("- ").strip())

                query = " ".join(query_parts)[:500]
                max_results = project.config.get("max_search_results", 5)
                results = store.search(query, project_id=project.project_id, top_k=max_results)

                if results:
                    sections.append("## Recent Relevant Memories\n")
                    for i, r in enumerate(results, 1):
                        ntype = r.get("node_type", "unknown").upper()
                        sim = r.get("similarity", 0)
                        pivot = " (PIVOT)" if r.get("is_pivot") else ""
                        sections.append(
                            f"{i}. **[{ntype}{pivot}]** (relevance: {sim:.0%})\n"
                            f"   {r['summary']}\n"
                        )

                sections.append(
                    f"\n*Cognitive memory: {node_count} reasoning nodes from "
                    f"{project.session_count} past sessions.*"
                )
        except Exception as e:
            sections.append(f"*Cognitive memory available but retrieval failed: {e}*")

    if not sections:
        return (
            f"Cognitive memory initialized for {project.name} "
            f"but no sessions ingested yet."
        )

    return "\n\n".join(sections)


def session_stop_hook(project_dir: Path | None = None) -> dict:
    """Session-stop hook: ingest the just-completed session.

    Returns a dict with ingestion stats, or error info.
    """
    project = discover_project(project_dir)
    if project is None:
        return {"status": "skipped", "reason": "no .cmm/ folder found"}

    if not project.config.get("auto_ingest", True):
        return {"status": "skipped", "reason": "auto_ingest disabled in config"}

    # Find the latest session transcript
    session_file = _find_latest_session(project)
    if session_file is None:
        return {"status": "skipped", "reason": "no session transcript found"}

    # Check if this session was already ingested (compare against last_session)
    if project.last_session and session_file.stem == project.last_session:
        return {"status": "skipped", "reason": "session already ingested"}

    store_path = _get_store_path(project)

    try:
        from src.ingestion import ClaudeCodeParser
        from src.extraction.warm_extractor import WarmExtractor
        from src.store.vector_store import MemoryStore
        from src.compression.dedup import SemanticDeduplicator
        from src.schemas.reasoning import ReasoningDAG

        # Parse
        parser = ClaudeCodeParser()
        session = parser.parse_file(session_file)

        # Extract (warm tier — fast, no API key)
        extractor = WarmExtractor()
        dag = extractor.extract(session)

        # Deduplicate and store
        store = MemoryStore(persist_dir=store_path)
        dedup = SemanticDeduplicator(store)
        result = dedup.deduplicate(dag.nodes, project.project_id, session.session_id)

        dag_to_store = ReasoningDAG(
            session_id=dag.session_id,
            nodes=result.stored,
            edges=dag.edges,
            pivot_nodes=dag.pivot_nodes,
            noise_ratio=dag.noise_ratio,
        )
        stored_count = store.store_dag(dag_to_store, project.project_id)

        # Update manifest
        project.update_session(session_file.stem)

        # Run post-session evaluation
        evaluation = None
        try:
            from src.evaluation.logger import InteractionLogger
            from src.evaluation.analyzer import SessionAnalyzer

            logger = InteractionLogger()
            analyzer = SessionAnalyzer(logger, store_path=store_path)
            evaluation = analyzer.analyze(
                session=session,
                dag=dag,
                project_id=project.project_id,
                session_id=session_file.stem,
            )
        except Exception:
            pass  # evaluation is optional — never block ingestion

        # Run profile quality checks (no API key needed)
        profile_quality = None
        try:
            profile = store.get_profile(project.project_id)
            if profile:
                from src.evaluation.profile_quality import run_quality_checks
                total_sessions = project.session_count
                profile_quality = run_quality_checks(
                    profile=profile,
                    project_dir=project.project_dir,
                    total_sessions=total_sessions,
                )
        except Exception:
            pass  # never block ingestion

        return {
            "status": "ingested",
            "session_file": str(session_file),
            "messages": len(session.messages),
            "nodes_extracted": len(dag.nodes),
            "nodes_stored": stored_count,
            "nodes_merged": len(result.merged),
            "nodes_dropped": len(result.dropped),
            "evaluation": evaluation,
            "profile_quality": profile_quality,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
