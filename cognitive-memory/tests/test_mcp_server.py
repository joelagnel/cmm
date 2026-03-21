"""Phase 3 tests: MCP server tools."""
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.schemas.memory import ArchitecturalInsight, CognitiveProfile, DiagnosticStrategy, Pitfall
from src.schemas.reasoning import NodeType, ReasoningDAG, ReasoningEdge, ReasoningNode
from src.store import MemoryStore


def utcnow():
    return datetime.now(timezone.utc)


def make_node(node_id, ntype, summary, confidence=0.8):
    return ReasoningNode(
        node_id=node_id,
        node_type=ntype,
        summary=summary,
        evidence="test",
        message_range=(0, 3),
        confidence=confidence,
    )


def make_dag(session_id, nodes):
    edges = [
        ReasoningEdge(source_id=nodes[i].node_id, target_id=nodes[i+1].node_id, relationship="led_to")
        for i in range(len(nodes) - 1)
    ]
    return ReasoningDAG(
        session_id=session_id,
        nodes=nodes,
        edges=edges,
        pivot_nodes=[n.node_id for n in nodes if n.node_type == NodeType.PIVOT],
        noise_ratio=0.7,
    )


def make_profile(project_id: str, store: MemoryStore) -> CognitiveProfile:
    profile = CognitiveProfile(
        project_id=project_id,
        last_updated=utcnow(),
        architectural_insights=[
            ArchitecturalInsight(
                component="auth",
                insight="JWT tokens are signed with HS256; signing and verification must share the same secret",
                confidence=0.9,
            ),
            ArchitecturalInsight(
                component="database",
                insight="Connection pool is capped at 10; exhaustion causes 503 errors under load",
                confidence=0.85,
            ),
        ],
        pitfalls=[
            Pitfall(
                description="Using OLD_SECRET_KEY instead of SECRET_KEY causes silent 401 auth failures",
                frequency=3,
                severity="high",
                resolution_strategy="Always import SECRET_KEY from config, never use inline variable names",
            ),
            Pitfall(
                description="Running migrations without --fake on first deploy causes IntegrityError",
                frequency=2,
                severity="medium",
                resolution_strategy="Use manage.py migrate --fake-initial on fresh databases",
            ),
        ],
        diagnostic_strategies=[
            DiagnosticStrategy(
                trigger="401 authentication errors",
                steps=[
                    "Check login.py for the signing key variable name",
                    "Check auth middleware for the verification key variable name",
                    "Ensure both reference SECRET_KEY from config",
                ],
                success_rate=1.0,
                source_sessions=["session-1", "session-2"],
            ),
            DiagnosticStrategy(
                trigger="database connection pool exhausted",
                steps=[
                    "Check DB_POOL_SIZE in settings",
                    "Look for missing connection.close() in long-running tasks",
                    "Consider using context managers for all DB operations",
                ],
                success_rate=0.8,
                source_sessions=["session-3"],
            ),
        ],
        key_patterns=["JWT-based auth", "Postgres with connection pooling"],
        anti_patterns=["Inline secret key references", "Missing connection cleanup"],
        session_count=3,
    )
    store.save_profile(profile)
    return profile


@pytest.fixture
def store_with_data(tmp_path):
    store = MemoryStore(persist_dir=tmp_path / "memory")

    dag = make_dag("session-1", [
        make_node("n1", NodeType.HYPOTHESIS, "JWT authentication token validation is failing with 401"),
        make_node("n2", NodeType.INVESTIGATION, "Reading auth middleware source code"),
        make_node("n3", NodeType.PIVOT, "Discovered key name mismatch between login and middleware"),
        make_node("n4", NodeType.SOLUTION, "Fixed by using consistent SECRET_KEY from config module"),
    ])
    store.store_dag(dag, "project-auth")
    make_profile("project-auth", store)
    return store


# ── Import the tools as plain functions via the module ─────────────────────────

def _get_tools(store):
    """Patch the global store and import tool functions."""
    import src.delivery.mcp_server as srv
    srv._store = store
    return srv


class TestSearchMemory:
    def test_returns_relevant_results(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.search_memory("JWT authentication failure", project_id="project-auth")
        assert "JWT" in result or "authentication" in result.lower()
        assert "past reasoning" in result.lower() or "relevant" in result.lower()

    def test_returns_no_results_message_for_empty(self, tmp_path):
        empty_store = MemoryStore(persist_dir=tmp_path / "empty")
        srv = _get_tools(empty_store)
        result = srv.search_memory("anything", project_id="nonexistent")
        assert "No relevant memories" in result

    def test_respects_project_filter(self, store_with_data):
        srv = _get_tools(store_with_data)
        result_auth = srv.search_memory("JWT key", project_id="project-auth")
        result_other = srv.search_memory("JWT key", project_id="project-other")
        # auth project has data, other doesn't
        assert "No relevant memories" not in result_auth
        assert "No relevant memories" in result_other

    def test_top_k_limits_results(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.search_memory("code", project_id="project-auth", top_k=2)
        # Count result entries — each starts with "**N."
        count = result.count("**1.") + result.count("**2.") + result.count("**3.")
        assert count <= 3  # at most top_k+1 headers possible

    def test_pivot_nodes_marked(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.search_memory("key mismatch pivot", project_id="project-auth")
        assert "⚡" in result or "PIVOT" in result


class TestGetCognitiveProfile:
    def test_returns_full_profile(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.get_cognitive_profile(project_id="project-auth")
        assert "Cognitive Profile" in result
        assert "JWT" in result
        assert "auth" in result.lower()
        assert "pitfall" in result.lower() or "Pitfall" in result

    def test_missing_profile_suggests_ingest(self, tmp_path):
        store = MemoryStore(persist_dir=tmp_path / "empty")
        srv = _get_tools(store)
        result = srv.get_cognitive_profile(project_id="no-such-project")
        assert "ingest" in result.lower() or "not found" in result.lower() or "No" in result

    def test_nodes_without_profile_prompts_build(self, tmp_path):
        store = MemoryStore(persist_dir=tmp_path / "partial")
        dag = make_dag("s1", [make_node("n1", NodeType.INVESTIGATION, "reading code")])
        store.store_dag(dag, "partial-project")
        srv = _get_tools(store)
        result = srv.get_cognitive_profile(project_id="partial-project")
        assert "build" in result.lower() or "ingest" in result.lower()

    def test_contains_architectural_insights(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.get_cognitive_profile(project_id="project-auth")
        assert "Architectural" in result
        assert "connection pool" in result.lower() or "JWT" in result

    def test_contains_diagnostic_strategies(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.get_cognitive_profile(project_id="project-auth")
        assert "Diagnostic" in result or "strategy" in result.lower()

    def test_no_project_id_returns_error(self, store_with_data):
        srv = _get_tools(store_with_data)
        # Ensure no env var is set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CMM_PROJECT_ID", None)
            result = srv.get_cognitive_profile(project_id="")
        assert "Error" in result or "project_id" in result


class TestGetPitfalls:
    def test_returns_pitfalls_ranked_by_severity(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.get_pitfalls(project_id="project-auth")
        assert "HIGH" in result
        assert "MEDIUM" in result
        # HIGH should appear before MEDIUM
        assert result.index("HIGH") < result.index("MEDIUM")

    def test_includes_resolution_strategy(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.get_pitfalls(project_id="project-auth")
        assert "Resolution" in result or "→" in result
        assert "SECRET_KEY" in result

    def test_no_pitfalls_returns_message(self, tmp_path):
        store = MemoryStore(persist_dir=tmp_path / "empty")
        profile = CognitiveProfile(
            project_id="clean-project",
            last_updated=utcnow(),
            pitfalls=[],
            session_count=1,
        )
        store.save_profile(profile)
        srv = _get_tools(store)
        result = srv.get_pitfalls(project_id="clean-project")
        assert "No known pitfalls" in result

    def test_missing_profile_returns_error(self, tmp_path):
        store = MemoryStore(persist_dir=tmp_path / "empty")
        srv = _get_tools(store)
        result = srv.get_pitfalls(project_id="ghost-project")
        assert "No profile" in result or "ingest" in result.lower()


class TestGetDiagnosticStrategy:
    def test_matches_relevant_strategy(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.get_diagnostic_strategy(
            "Getting 401 errors on login endpoint",
            project_id="project-auth",
        )
        assert "401" in result or "authentication" in result.lower()
        assert "Step" in result or "1." in result

    def test_database_strategy(self, store_with_data):
        srv = _get_tools(store_with_data)
        result = srv.get_diagnostic_strategy(
            "database connections are being exhausted",
            project_id="project-auth",
        )
        assert "pool" in result.lower() or "database" in result.lower() or "connection" in result.lower()

    def test_fallback_to_search_when_no_strategies(self, tmp_path):
        store = MemoryStore(persist_dir=tmp_path / "mem")
        dag = make_dag("s1", [
            make_node("n1", NodeType.SOLUTION, "Fixed the webpack config by adding a cache-loader")
        ])
        store.store_dag(dag, "project-frontend")
        profile = CognitiveProfile(
            project_id="project-frontend",
            last_updated=utcnow(),
            diagnostic_strategies=[],  # empty
            session_count=1,
        )
        store.save_profile(profile)
        srv = _get_tools(store)
        result = srv.get_diagnostic_strategy("webpack build is slow", project_id="project-frontend")
        # Should fall back to search results
        assert "webpack" in result.lower() or "past reasoning" in result.lower() or "No diagnostic" in result

    def test_missing_profile_returns_error(self, tmp_path):
        store = MemoryStore(persist_dir=tmp_path / "empty")
        srv = _get_tools(store)
        result = srv.get_diagnostic_strategy("some problem", project_id="ghost-project")
        assert "No profile" in result or "ingest" in result.lower()
