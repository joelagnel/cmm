"""Push/pull synchronization between local and shared MemoryStores.

Push: takes locally-generated nodes that haven't been pushed yet, copies
their embeddings (no re-embedding) into the shared store's STAGING
collection, marks them locally as pushed.

Pull: fetches all approved nodes for the project from the shared store's
MAIN collection (plus all approved team-scope nodes regardless of project),
deduplicates against existing local nodes, and inserts them locally.

Sync events are logged to a SQLite database at data/sync/sync.db. The log
is auditable and survives across runs.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..compression.dedup import SemanticDeduplicator
from ..store.vector_store import MemoryStore


_DEFAULT_LOG_PATH = Path(__file__).parent.parent.parent / "data" / "sync" / "sync.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sync_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      TEXT NOT NULL,
    action          TEXT NOT NULL,        -- 'push', 'pull', 'approve', 'reject'
    timestamp       TEXT NOT NULL,
    actor           TEXT,                  -- developer name or reviewer
    count           INTEGER DEFAULT 0,
    detail          TEXT                   -- JSON or short description
);

CREATE INDEX IF NOT EXISTS idx_sync_log_project ON sync_log(project_id);
CREATE INDEX IF NOT EXISTS idx_sync_log_action ON sync_log(action);
"""


# ── Result types ────────────────────────────────────────────────────────


@dataclass
class PushResult:
    project_id: str
    pushed: int = 0
    already_pushed: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""

    @property
    def summary(self) -> str:
        msg = f"Pushed {self.pushed} nodes to staging for review"
        if self.already_pushed:
            msg += f" ({self.already_pushed} already pushed)"
        return msg


@dataclass
class PullResult:
    project_id: str
    pulled_project: int = 0
    pulled_team: int = 0
    deduped: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""

    @property
    def summary(self) -> str:
        return (
            f"Pulled {self.pulled_project} project nodes "
            f"+ {self.pulled_team} team nodes "
            f"({self.deduped} duplicates skipped)"
        )


# ── Sync log ────────────────────────────────────────────────────────────


class SyncLog:
    """SQLite-backed audit log for all sync events.

    Designed to be fail-safe: if the DB is missing or unwritable,
    sync operations still succeed.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_LOG_PATH
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # fail-safe: bad path → all subsequent ops no-op
        self._init_db()

    def _init_db(self):
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except Exception:
            pass

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        project_id: str,
        action: str,
        count: int = 0,
        actor: str = "",
        detail: str = "",
    ) -> None:
        """Append an event to the sync log. Fails silently."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO sync_log
                       (project_id, action, timestamp, actor, count, detail)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        project_id,
                        action,
                        datetime.now(timezone.utc).isoformat(),
                        actor,
                        count,
                        detail,
                    ),
                )
        except Exception:
            pass

    def last_event(self, project_id: str, action: str) -> dict | None:
        """Most recent event of a given action for a project."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """SELECT * FROM sync_log
                       WHERE project_id = ? AND action = ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    (project_id, action),
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    def history(self, project_id: str | None = None, limit: int = 50) -> list[dict]:
        try:
            with self._connect() as conn:
                if project_id:
                    rows = conn.execute(
                        """SELECT * FROM sync_log
                           WHERE project_id = ?
                           ORDER BY timestamp DESC LIMIT ?""",
                        (project_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM sync_log
                           ORDER BY timestamp DESC LIMIT ?""",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []


# ── Syncer ──────────────────────────────────────────────────────────────


class Syncer:
    """Coordinates push/pull between a local store and its shared counterpart."""

    def __init__(
        self,
        store: MemoryStore,
        log: SyncLog | None = None,
        developer: str | None = None,
    ):
        if not store.has_shared:
            raise RuntimeError(
                "Syncer requires a MemoryStore configured with shared_path. "
                "Initialize the store with shared_path=... first."
            )
        self.store = store
        self.log = log or SyncLog()
        self.developer = developer or os.environ.get("CMM_DEVELOPER_NAME", "")

    # ── Push ────────────────────────────────────────────────────────

    def push(self, project_id: str, dry_run: bool = False) -> PushResult:
        """Push all unpushed local nodes to the shared staging area."""
        result = PushResult(project_id=project_id)
        result.timestamp = datetime.now(timezone.utc).isoformat()

        unpushed = self.store.get_unpushed_nodes(project_id)
        if not unpushed:
            return result

        if dry_run:
            result.pushed = len(unpushed)
            return result

        try:
            count = self.store.stage_to_shared(unpushed, developer=self.developer)
            ids = [n["id"] for n in unpushed]
            self.store.mark_pushed(ids, when=result.timestamp)
            result.pushed = count
            self.log.record(
                project_id=project_id,
                action="push",
                count=count,
                actor=self.developer,
                detail=f"staged {count} nodes",
            )
        except Exception as e:
            result.errors.append(str(e))

        return result

    # ── Pull ────────────────────────────────────────────────────────

    def pull(self, project_id: str, include_team: bool = True) -> PullResult:
        """Pull approved nodes from the shared store into the local store.

        Deduplicates against existing local nodes by re-using the
        SemanticDeduplicator with the same threshold (0.85).
        """
        result = PullResult(project_id=project_id)
        result.timestamp = datetime.now(timezone.utc).isoformat()

        try:
            approved = self.store.list_approved_shared(
                project_id=project_id,
                include_team=include_team,
            )
        except Exception as e:
            result.errors.append(f"list_approved failed: {e}")
            return result

        if not approved:
            self.log.record(
                project_id=project_id,
                action="pull",
                count=0,
                actor=self.developer,
                detail="nothing to pull",
            )
            return result

        # Split by scope for reporting
        project_nodes = [n for n in approved if n.get("scope") != "team"]
        team_nodes = [n for n in approved if n.get("scope") == "team"]

        # Filter out anything we already have locally (by exact id match)
        existing = self.store.nodes_col_local.get(include=[])
        existing_ids = set(existing.get("ids", []))

        new_nodes = [n for n in approved if n["id"] not in existing_ids]
        result.deduped = len(approved) - len(new_nodes)

        if new_nodes:
            inserted = self.store.upsert_pulled_nodes(new_nodes, project_id=project_id)
            # Re-count by scope from new_nodes
            result.pulled_project = sum(1 for n in new_nodes if n.get("scope") != "team")
            result.pulled_team = sum(1 for n in new_nodes if n.get("scope") == "team")
        else:
            result.pulled_project = 0
            result.pulled_team = 0

        self.log.record(
            project_id=project_id,
            action="pull",
            count=result.pulled_project + result.pulled_team,
            actor=self.developer,
            detail=f"project={result.pulled_project} team={result.pulled_team} dedup={result.deduped}",
        )
        return result

    # ── Status ──────────────────────────────────────────────────────

    def status(self, project_id: str) -> dict[str, Any]:
        """Return a status snapshot for the CLI."""
        local_count = self.store.node_count(project_id)
        unpushed = len(self.store.get_unpushed_nodes(project_id))
        pending = len(self.store.list_pending_in_staging(project_id))

        shared_approved = 0
        if self.store.has_shared:
            try:
                shared_approved = len(self.store.list_approved_shared(
                    project_id=project_id, include_team=False
                ))
            except Exception:
                pass

        last_push = self.log.last_event(project_id, "push")
        last_pull = self.log.last_event(project_id, "pull")

        return {
            "project_id": project_id,
            "local_nodes": local_count,
            "unpushed_nodes": unpushed,
            "shared_approved": shared_approved,
            "pending_review": pending,
            "last_push": last_push.get("timestamp") if last_push else None,
            "last_pull": last_pull.get("timestamp") if last_pull else None,
            "developer": self.developer,
        }
