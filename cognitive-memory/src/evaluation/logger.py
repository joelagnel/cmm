"""Interaction logger — records every skill invocation to SQLite.

Non-intrusive: if logging fails, the retrieval still works.
Adds <5ms overhead per invocation (single SQLite INSERT).

Storage: data/eval/interactions.db
Table: invocations — one row per skill call during a session.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "eval" / "interactions.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS invocations (
    invocation_id   TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    skill           TEXT NOT NULL,
    query_text      TEXT,
    result_count    INTEGER DEFAULT 0,
    node_ids        TEXT,
    similarity_scores TEXT,
    response_time_ms REAL DEFAULT 0,
    raw_output_len  INTEGER DEFAULT 0,
    estimated_message_index INTEGER DEFAULT -1
);

CREATE TABLE IF NOT EXISTS session_evaluations (
    session_id              TEXT PRIMARY KEY,
    project_id              TEXT NOT NULL,
    evaluated_at            TEXT NOT NULL,
    total_invocations       INTEGER DEFAULT 0,
    memory_used_at_start    INTEGER DEFAULT 0,
    errors_encountered      INTEGER DEFAULT 0,
    errors_resolved_with_memory INTEGER DEFAULT 0,
    pitfalls_surfaced       INTEGER DEFAULT 0,
    pitfalls_avoided        INTEGER DEFAULT 0,
    pivots_after_retrieval  INTEGER DEFAULT 0,
    harmful_memory_count    INTEGER DEFAULT 0,
    total_dead_ends         INTEGER DEFAULT 0,
    total_pivots            INTEGER DEFAULT 0,
    messages_to_first_solution INTEGER DEFAULT 0,
    total_messages          INTEGER DEFAULT 0,
    total_nodes             INTEGER DEFAULT 0,
    duration_seconds        REAL DEFAULT 0
);
"""


class InteractionLogger:
    """Logs skill invocations to SQLite. Designed to be fail-safe."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
                # Migrate: add estimated_message_index if missing (pre-existing DBs)
                try:
                    conn.execute(
                        "ALTER TABLE invocations ADD COLUMN estimated_message_index INTEGER DEFAULT -1"
                    )
                except Exception:
                    pass  # column already exists
        except Exception:
            pass  # fail silently — logging should never break retrieval

    @contextmanager
    def _connect(self):
        """Context manager for SQLite connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Derive session ID ─────────────────────────────────────────

    @staticmethod
    def get_session_id() -> str:
        """Get current session ID from env or derive from latest JSONL file."""
        # Check if set explicitly
        sid = os.environ.get("CMM_SESSION_ID")
        if sid:
            return sid

        # Try to find the most recently modified JSONL in ~/.claude/projects/
        try:
            claude_projects = Path.home() / ".claude" / "projects"
            if claude_projects.exists():
                latest = None
                latest_mtime = 0.0
                for project_dir in claude_projects.iterdir():
                    if not project_dir.is_dir():
                        continue
                    for f in project_dir.glob("*.jsonl"):
                        mtime = f.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest = f
                if latest:
                    return latest.stem
        except Exception:
            pass

        # Fallback: generate a session-scoped ID
        return f"unknown-{uuid.uuid4().hex[:8]}"

    # ── Log an invocation ─────────────────────────────────────────

    @staticmethod
    def count_session_messages(session_id: str | None = None) -> int:
        """Count how many lines the current session JSONL has.

        This gives us the approximate message index at the time of
        invocation — far more accurate than the old even-distribution guess.
        Returns -1 if the file can't be read.
        """
        try:
            sid = session_id
            if not sid:
                sid = os.environ.get("CMM_SESSION_ID")
            if not sid:
                # Find the latest session file
                claude_projects = Path.home() / ".claude" / "projects"
                if claude_projects.exists():
                    latest = None
                    latest_mtime = 0.0
                    for d in claude_projects.iterdir():
                        if not d.is_dir():
                            continue
                        for f in d.glob("*.jsonl"):
                            mtime = f.stat().st_mtime
                            if mtime > latest_mtime:
                                latest_mtime = mtime
                                latest = f
                    if latest:
                        sid = latest.stem
                        # Count lines in this file
                        return sum(1 for _ in latest.open())
                return -1

            # sid is known — search for it
            claude_projects = Path.home() / ".claude" / "projects"
            if claude_projects.exists():
                for d in claude_projects.iterdir():
                    if not d.is_dir():
                        continue
                    candidate = d / f"{sid}.jsonl"
                    if candidate.exists():
                        return sum(1 for _ in candidate.open())
        except Exception:
            pass
        return -1

    def log_invocation(
        self,
        skill: str,
        project_id: str,
        query_text: str | None = None,
        results: list[dict[str, Any]] | None = None,
        response_time_ms: float = 0.0,
        raw_output_len: int = 0,
        estimated_message_index: int = -1,
    ):
        """Log a single skill invocation. Fails silently.

        Args:
            estimated_message_index: the current message count in the session
                JSONL at invocation time. If -1, the caller didn't measure it.
        """
        try:
            session_id = self.get_session_id()
            invocation_id = str(uuid.uuid4())
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

            node_ids = []
            similarity_scores = []
            if results:
                for r in results:
                    nid = r.get("node_id", "")
                    if nid:
                        node_ids.append(nid)
                    sim = r.get("similarity")
                    if sim is not None:
                        similarity_scores.append(round(sim, 4))

            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO invocations
                       (invocation_id, session_id, project_id, timestamp,
                        skill, query_text, result_count, node_ids,
                        similarity_scores, response_time_ms, raw_output_len,
                        estimated_message_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        invocation_id,
                        session_id,
                        project_id,
                        timestamp,
                        skill,
                        query_text,
                        len(results) if results else 0,
                        json.dumps(node_ids),
                        json.dumps(similarity_scores),
                        response_time_ms,
                        raw_output_len,
                        estimated_message_index,
                    ),
                )
        except Exception:
            pass  # never break retrieval

    # ── Save session evaluation ───────────────────────────────────

    def save_session_evaluation(self, evaluation: dict[str, Any]):
        """Save a post-session evaluation. Fails silently."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO session_evaluations
                       (session_id, project_id, evaluated_at,
                        total_invocations, memory_used_at_start,
                        errors_encountered, errors_resolved_with_memory,
                        pitfalls_surfaced, pitfalls_avoided,
                        pivots_after_retrieval, harmful_memory_count,
                        total_dead_ends, total_pivots,
                        messages_to_first_solution, total_messages,
                        total_nodes, duration_seconds)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        evaluation["session_id"],
                        evaluation["project_id"],
                        evaluation.get("evaluated_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                        evaluation.get("total_invocations", 0),
                        evaluation.get("memory_used_at_start", 0),
                        evaluation.get("errors_encountered", 0),
                        evaluation.get("errors_resolved_with_memory", 0),
                        evaluation.get("pitfalls_surfaced", 0),
                        evaluation.get("pitfalls_avoided", 0),
                        evaluation.get("pivots_after_retrieval", 0),
                        evaluation.get("harmful_memory_count", 0),
                        evaluation.get("total_dead_ends", 0),
                        evaluation.get("total_pivots", 0),
                        evaluation.get("messages_to_first_solution", 0),
                        evaluation.get("total_messages", 0),
                        evaluation.get("total_nodes", 0),
                        evaluation.get("duration_seconds", 0),
                    ),
                )
        except Exception:
            pass

    # ── Query helpers ─────────────────────────────────────────────

    def get_invocations(
        self,
        session_id: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve logged invocations."""
        try:
            with self._connect() as conn:
                where_clauses = []
                params = []
                if session_id:
                    where_clauses.append("session_id = ?")
                    params.append(session_id)
                if project_id:
                    where_clauses.append("project_id = ?")
                    params.append(project_id)

                where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                rows = conn.execute(
                    f"SELECT * FROM invocations {where} ORDER BY timestamp DESC LIMIT ?",
                    params + [limit],
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_session_evaluations(
        self,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve session evaluations."""
        try:
            with self._connect() as conn:
                if project_id:
                    rows = conn.execute(
                        "SELECT * FROM session_evaluations WHERE project_id = ? ORDER BY evaluated_at DESC LIMIT ?",
                        (project_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM session_evaluations ORDER BY evaluated_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_stats(self, project_id: str | None = None) -> dict:
        """Get aggregate statistics."""
        try:
            with self._connect() as conn:
                where = "WHERE project_id = ?" if project_id else ""
                params = [project_id] if project_id else []

                inv_count = conn.execute(
                    f"SELECT COUNT(*) FROM invocations {where}", params
                ).fetchone()[0]

                sess_count = conn.execute(
                    f"SELECT COUNT(*) FROM session_evaluations {where}", params
                ).fetchone()[0]

                # Average similarity scores
                avg_sim = None
                rows = conn.execute(
                    f"SELECT similarity_scores FROM invocations {where}", params
                ).fetchall()
                all_sims = []
                for row in rows:
                    scores = json.loads(row[0]) if row[0] else []
                    all_sims.extend(scores)
                if all_sims:
                    avg_sim = sum(all_sims) / len(all_sims)

                # Average response time
                avg_rt = conn.execute(
                    f"SELECT AVG(response_time_ms) FROM invocations {where}", params
                ).fetchone()[0]

                # Session evaluation aggregates
                eval_stats = {}
                if sess_count > 0:
                    row = conn.execute(
                        f"""SELECT
                            AVG(total_dead_ends) as avg_dead_ends,
                            AVG(total_pivots) as avg_pivots,
                            AVG(messages_to_first_solution) as avg_msgs_to_solution,
                            SUM(pitfalls_surfaced) as total_pitfalls_surfaced,
                            SUM(pitfalls_avoided) as total_pitfalls_avoided,
                            SUM(errors_resolved_with_memory) as total_errors_resolved,
                            SUM(memory_used_at_start) as sessions_with_early_memory
                        FROM session_evaluations {where}""",
                        params,
                    ).fetchone()
                    eval_stats = dict(row) if row else {}

                return {
                    "total_invocations": inv_count,
                    "total_sessions_evaluated": sess_count,
                    "avg_similarity": round(avg_sim, 3) if avg_sim else None,
                    "avg_response_time_ms": round(avg_rt, 1) if avg_rt else None,
                    **eval_stats,
                }
        except Exception:
            return {}
