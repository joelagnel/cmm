"""Tests for the evaluation framework — logger, analyzer, dashboard."""
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.logger import InteractionLogger
from src.evaluation.analyzer import SessionAnalyzer
from src.schemas.reasoning import NodeType, ReasoningDAG, ReasoningEdge, ReasoningNode
from src.schemas.session import NormalizedSession, SessionMessage, MessageRole


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def eval_db(tmp_path):
    """Create a logger with a temp database."""
    return InteractionLogger(db_path=tmp_path / "test_eval.db")


def _make_session(messages: list[tuple[str, str]], session_id="test-session") -> NormalizedSession:
    msgs = []
    for role_str, content in messages:
        msgs.append(SessionMessage(
            role=MessageRole(role_str),
            content=content,
            timestamp=datetime.now(timezone.utc),
        ))
    return NormalizedSession(
        session_id=session_id,
        source_platform="test",
        started_at=datetime.now(timezone.utc),
        messages=msgs,
    )


def _make_dag(session_id, node_specs) -> ReasoningDAG:
    """node_specs: list of (node_type, summary, msg_start, msg_end)"""
    nodes = []
    for i, (ntype, summary, ms, me) in enumerate(node_specs):
        nodes.append(ReasoningNode(
            node_id=f"n-{i:03d}",
            node_type=ntype,
            summary=summary,
            evidence="test",
            message_range=(ms, me),
            confidence=0.7,
        ))
    edges = [
        ReasoningEdge(source_id=nodes[i].node_id, target_id=nodes[i + 1].node_id, relationship="led_to")
        for i in range(len(nodes) - 1)
    ]
    pivots = [n.node_id for n in nodes if n.node_type in (NodeType.PIVOT, NodeType.DEAD_END)]
    return ReasoningDAG(
        session_id=session_id,
        nodes=nodes,
        edges=edges,
        pivot_nodes=pivots,
        noise_ratio=0.5,
    )


# ── InteractionLogger Tests ──────────────────────────────────────


class TestInteractionLogger:

    def test_log_and_retrieve_invocation(self, eval_db):
        """Basic log and retrieve cycle."""
        eval_db.log_invocation(
            skill="search-memory",
            project_id="test-project",
            query_text="database timeout",
            results=[
                {"node_id": "n-001", "similarity": 0.85, "summary": "DB timeout fix"},
                {"node_id": "n-002", "similarity": 0.72, "summary": "Connection pool"},
            ],
            response_time_ms=42.5,
            raw_output_len=200,
        )

        invocations = eval_db.get_invocations(project_id="test-project")
        assert len(invocations) == 1

        inv = invocations[0]
        assert inv["skill"] == "search-memory"
        assert inv["query_text"] == "database timeout"
        assert inv["result_count"] == 2
        assert inv["response_time_ms"] == 42.5
        assert json.loads(inv["node_ids"]) == ["n-001", "n-002"]
        assert json.loads(inv["similarity_scores"]) == [0.85, 0.72]

    def test_log_profile_invocation(self, eval_db):
        """Profile invocations log with no query or results."""
        eval_db.log_invocation(
            skill="cognitive-profile",
            project_id="test-project",
            response_time_ms=15.0,
            raw_output_len=500,
        )

        invocations = eval_db.get_invocations()
        assert len(invocations) == 1
        assert invocations[0]["skill"] == "cognitive-profile"
        assert invocations[0]["query_text"] is None
        assert invocations[0]["result_count"] == 0

    def test_multiple_invocations(self, eval_db):
        """Multiple invocations are tracked separately."""
        for i in range(5):
            eval_db.log_invocation(
                skill="search-memory",
                project_id="test-project",
                query_text=f"query {i}",
                results=[{"node_id": f"n-{i}", "similarity": 0.8}],
                response_time_ms=10.0 + i,
            )

        invocations = eval_db.get_invocations(project_id="test-project")
        assert len(invocations) == 5

    def test_save_and_retrieve_session_evaluation(self, eval_db):
        """Session evaluations are persisted and retrievable."""
        eval_db.save_session_evaluation({
            "session_id": "sess-001",
            "project_id": "test-project",
            "total_invocations": 3,
            "memory_used_at_start": 1,
            "errors_encountered": 2,
            "errors_resolved_with_memory": 1,
            "pitfalls_surfaced": 4,
            "pitfalls_avoided": 3,
            "pivots_after_retrieval": 1,
            "total_dead_ends": 2,
            "total_pivots": 1,
            "messages_to_first_solution": 15,
            "total_messages": 30,
            "total_nodes": 8,
            "duration_seconds": 120.0,
        })

        evals = eval_db.get_session_evaluations(project_id="test-project")
        assert len(evals) == 1
        assert evals[0]["pitfalls_avoided"] == 3
        assert evals[0]["errors_resolved_with_memory"] == 1

    def test_get_stats(self, eval_db):
        """Aggregate stats are computed correctly."""
        eval_db.log_invocation(
            skill="search-memory",
            project_id="proj-a",
            query_text="test",
            results=[{"node_id": "n1", "similarity": 0.9}],
            response_time_ms=30.0,
        )
        eval_db.log_invocation(
            skill="pitfalls",
            project_id="proj-a",
            response_time_ms=10.0,
        )

        stats = eval_db.get_stats(project_id="proj-a")
        assert stats["total_invocations"] == 2
        assert stats["avg_response_time_ms"] == 20.0
        assert stats["avg_similarity"] == 0.9

    def test_logging_failure_is_silent(self, tmp_path):
        """If the DB path is invalid, logging doesn't raise."""
        logger = InteractionLogger(db_path=tmp_path / "readonly" / "test.db")
        # This would fail if not handled — readonly dir doesn't exist
        # But the logger should handle it silently
        # (Actually mkdir will create it, so let's test with a truly bad path)
        logger._db_path = Path("/dev/null/impossible.db")
        # Should not raise
        logger.log_invocation(
            skill="test",
            project_id="test",
        )


# ── SessionAnalyzer Tests ─────────────────────────────────────────


class TestSessionAnalyzer:

    def test_analyze_empty_session(self, eval_db):
        """Analyze a session with no invocations."""
        session = _make_session([
            ("user", "hello"),
            ("assistant", "hi there"),
        ])
        dag = _make_dag("test-session", [
            (NodeType.CONTEXT_LOAD, "Greeting", 0, 2),
        ])

        analyzer = SessionAnalyzer(eval_db)
        result = analyzer.analyze(session, dag, "test-project")

        assert result["session_id"] == "test-session"
        assert result["total_invocations"] == 0
        assert result["total_dead_ends"] == 0
        assert result["total_messages"] == 2

    def test_analyze_session_with_dead_ends_and_solutions(self, eval_db):
        """Session stats correctly count dead ends and solutions."""
        session = _make_session([
            ("user", "fix the bug"),
            ("assistant", "trying approach A"),
            ("assistant", "approach A failed"),
            ("assistant", "trying approach B"),
            ("assistant", "fixed it"),
        ])
        dag = _make_dag("test-session", [
            (NodeType.HYPOTHESIS, "Trying approach A", 0, 2),
            (NodeType.DEAD_END, "Approach A failed", 2, 3),
            (NodeType.PIVOT, "Switching to approach B", 3, 4),
            (NodeType.SOLUTION, "Fixed the bug", 4, 5),
        ])

        analyzer = SessionAnalyzer(eval_db)
        result = analyzer.analyze(session, dag, "test-project")

        assert result["total_dead_ends"] == 1
        assert result["total_pivots"] == 1
        assert result["messages_to_first_solution"] == 4

    def test_analyze_detects_early_memory_use(self, eval_db):
        """Detects when cognitive-profile was called at session start."""
        # Log a profile invocation
        eval_db.log_invocation(
            skill="cognitive-profile",
            project_id="test-project",
        )
        # Manually set the session_id to match
        import sqlite3
        with eval_db._connect() as conn:
            conn.execute(
                "UPDATE invocations SET session_id = 'eval-session'"
            )

        session = _make_session([("user", "hi"), ("assistant", "hello")], session_id="eval-session")
        dag = _make_dag("eval-session", [
            (NodeType.CONTEXT_LOAD, "Starting session", 0, 2),
        ])

        analyzer = SessionAnalyzer(eval_db)
        result = analyzer.analyze(session, dag, "test-project", session_id="eval-session")

        assert result["memory_used_at_start"] == 1

    def test_compare_sessions(self, eval_db):
        """Session comparator produces valid comparison dict."""
        baseline_session = _make_session([
            ("user", "fix"), ("assistant", "trying"), ("assistant", "failed"),
            ("assistant", "trying again"), ("assistant", "failed again"),
            ("assistant", "ok now trying C"), ("assistant", "fixed"),
        ], session_id="baseline")
        baseline_dag = _make_dag("baseline", [
            (NodeType.HYPOTHESIS, "Approach A", 0, 2),
            (NodeType.DEAD_END, "Failed A", 2, 3),
            (NodeType.HYPOTHESIS, "Approach B", 3, 4),
            (NodeType.DEAD_END, "Failed B", 4, 5),
            (NodeType.PIVOT, "Switching to C", 5, 6),
            (NodeType.SOLUTION, "Fixed with C", 6, 7),
        ])

        assisted_session = _make_session([
            ("user", "fix"), ("assistant", "checking memory"),
            ("assistant", "using approach C directly"), ("assistant", "fixed"),
        ], session_id="assisted")
        assisted_dag = _make_dag("assisted", [
            (NodeType.CONTEXT_LOAD, "Loaded profile", 0, 1),
            (NodeType.INVESTIGATION, "Checking approach C", 1, 3),
            (NodeType.SOLUTION, "Fixed with C", 3, 4),
        ])

        analyzer = SessionAnalyzer(eval_db)
        comparison = analyzer.compare_sessions(
            baseline_session, baseline_dag,
            assisted_session, assisted_dag,
            "test-project",
        )

        assert comparison["baseline"]["dead_ends"] == 2
        assert comparison["assisted"]["dead_ends"] == 0
        assert comparison["baseline"]["pivots"] == 1
        assert comparison["assisted"]["pivots"] == 0
        assert comparison["reductions"]["dead_ends"] == 100.0
        assert comparison["baseline"]["total_messages"] > comparison["assisted"]["total_messages"]
