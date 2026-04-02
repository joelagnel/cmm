"""Phase 2 tests: vector store and deduplication."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from src.schemas.reasoning import NodeType, ReasoningDAG, ReasoningEdge, ReasoningNode
from src.schemas.memory import ArchitecturalInsight, CognitiveProfile, Pitfall
from src.store import MemoryStore
from src.compression import SemanticDeduplicator, DeduplicationResult


def utcnow():
    return datetime.now(timezone.utc)


def make_node(node_id: str, ntype: NodeType, summary: str, confidence: float = 0.8) -> ReasoningNode:
    return ReasoningNode(
        node_id=node_id,
        node_type=ntype,
        summary=summary,
        evidence="test evidence",
        message_range=(0, 3),
        confidence=confidence,
    )


def make_dag(session_id: str, nodes: list[ReasoningNode]) -> ReasoningDAG:
    edges = [
        ReasoningEdge(source_id=nodes[i].node_id, target_id=nodes[i+1].node_id, relationship="led_to")
        for i in range(len(nodes) - 1)
    ]
    pivots = [n.node_id for n in nodes if n.node_type == NodeType.PIVOT]
    return ReasoningDAG(
        session_id=session_id,
        nodes=nodes,
        edges=edges,
        pivot_nodes=pivots,
        noise_ratio=0.7,
    )


@pytest.fixture
def store(tmp_path):
    return MemoryStore(persist_dir=tmp_path / "memory")


# ── MemoryStore ──────────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_store_and_count(self, store):
        dag = make_dag("session-1", [
            make_node("n1", NodeType.HYPOTHESIS, "Agent suspects auth config issue"),
            make_node("n2", NodeType.INVESTIGATION, "Agent reads middleware code"),
            make_node("n3", NodeType.SOLUTION, "Agent fixes the JWT key mismatch"),
        ])
        stored = store.store_dag(dag, "project-auth")
        assert stored == 3
        assert store.node_count("project-auth") == 3

    def test_search_returns_relevant_results(self, store):
        dag = make_dag("session-1", [
            make_node("n1", NodeType.HYPOTHESIS, "JWT authentication token validation is failing"),
            make_node("n2", NodeType.INVESTIGATION, "Reading database connection pool configuration"),
            make_node("n3", NodeType.SOLUTION, "Fixed the JWT secret key mismatch between signing and verification"),
        ])
        store.store_dag(dag, "project-auth")

        results = store.search("JWT secret key problem", project_id="project-auth", top_k=3)
        assert len(results) >= 1
        # The JWT nodes should rank higher than the DB node
        top = results[0]
        assert "JWT" in top["summary"] or "jwt" in top["summary"].lower()

    def test_search_with_project_filter(self, store):
        dag_a = make_dag("s1", [make_node("n1", NodeType.INVESTIGATION, "Debugging React component lifecycle")])
        dag_b = make_dag("s2", [make_node("n1", NodeType.INVESTIGATION, "Debugging Django ORM query performance")])

        store.store_dag(dag_a, "project-frontend")
        store.store_dag(dag_b, "project-backend")

        results = store.search("React component", project_id="project-frontend", top_k=5)
        assert all(r["project_id"] == "project-frontend" for r in results)

    def test_search_across_multiple_sessions(self, store):
        for i in range(3):
            dag = make_dag(f"session-{i}", [
                make_node("n1", NodeType.DEAD_END if i % 2 == 0 else NodeType.INVESTIGATION,
                          f"Session {i}: database connection pool exhausted under load"),
            ])
            store.store_dag(dag, "project-x")

        results = store.search("database connection", project_id="project-x", top_k=5)
        assert len(results) == 3

    def test_persist_and_reload(self, tmp_path):
        store1 = MemoryStore(persist_dir=tmp_path / "memory")
        dag = make_dag("session-1", [
            make_node("n1", NodeType.SOLUTION, "Fixed the import cycle in utils module"),
        ])
        store1.store_dag(dag, "myproject")
        del store1

        # Re-open from disk
        store2 = MemoryStore(persist_dir=tmp_path / "memory")
        assert store2.node_count("myproject") == 1
        results = store2.search("import cycle", project_id="myproject")
        assert len(results) == 1

    def test_save_and_get_profile(self, store):
        profile = CognitiveProfile(
            project_id="myproject",
            last_updated=utcnow(),
            architectural_insights=[
                ArchitecturalInsight(component="auth", insight="JWT keys must match", confidence=0.9)
            ],
            pitfalls=[Pitfall(description="Missing env vars in production", severity="high")],
            session_count=3,
        )
        store.save_profile(profile)
        retrieved = store.get_profile("myproject")
        assert retrieved is not None
        assert retrieved.project_id == "myproject"
        assert len(retrieved.architectural_insights) == 1
        assert retrieved.pitfalls[0].severity == "high"

    def test_get_profile_missing_returns_none(self, store):
        assert store.get_profile("nonexistent-project") is None

    def test_list_projects(self, store):
        store.store_dag(
            make_dag("s1", [make_node("n1", NodeType.CONTEXT_LOAD, "Reading codebase")]),
            "project-alpha",
        )
        store.store_dag(
            make_dag("s2", [make_node("n1", NodeType.CONTEXT_LOAD, "Reading docs")]),
            "project-beta",
        )
        projects = store.list_projects()
        assert "project-alpha" in projects
        assert "project-beta" in projects


# ── SemanticDeduplicator ─────────────────────────────────────────────────────

class TestSemanticDeduplicator:
    def test_all_new_when_store_empty(self, store):
        dedup = SemanticDeduplicator(store, similarity_threshold=0.85)
        nodes = [
            make_node("n1", NodeType.HYPOTHESIS, "Agent suspects the cache is stale"),
            make_node("n2", NodeType.SOLUTION, "Cleared Redis cache, problem resolved"),
        ]
        result = dedup.deduplicate(nodes, "project-cache", "session-1")
        assert len(result.stored) == 2
        assert len(result.dropped) == 0

    def test_near_duplicate_dropped_when_lower_confidence(self, store):
        # Store original node
        original = make_dag("session-1", [
            make_node("n1", NodeType.SOLUTION,
                      "Fixed the JWT signing secret key mismatch in the auth middleware config",
                      confidence=0.9),
        ])
        store.store_dag(original, "project-jwt")

        # Try to store near-duplicate with lower confidence (same words, diff order)
        dedup = SemanticDeduplicator(store, similarity_threshold=0.70)
        duplicate = make_node(
            "n1", NodeType.SOLUTION,
            "Fixed the JWT signing secret key mismatch in the auth middleware configuration",
            confidence=0.6,
        )
        result = dedup.deduplicate([duplicate], "project-jwt", "session-2")
        assert len(result.dropped) == 1
        assert len(result.stored) == 0

    def test_near_duplicate_replaces_when_higher_confidence(self, store):
        # Store original with low confidence
        original = make_dag("session-1", [
            make_node("n1", NodeType.SOLUTION,
                      "Fixed the JWT signing secret key mismatch in the auth middleware config",
                      confidence=0.4),
        ])
        store.store_dag(original, "project-jwt")

        dedup = SemanticDeduplicator(store, similarity_threshold=0.70)
        better = make_node(
            "n2", NodeType.SOLUTION,
            "Fixed the JWT signing secret key mismatch in the auth middleware configuration",
            confidence=0.95,
        )
        result = dedup.deduplicate([better], "project-jwt", "session-2")
        assert len(result.stored) == 1
        assert len(result.merged) == 1

    def test_distinct_nodes_all_stored(self, store):
        original = make_dag("session-1", [
            make_node("n1", NodeType.INVESTIGATION, "Checking webpack bundle size"),
        ])
        store.store_dag(original, "project-ui")

        dedup = SemanticDeduplicator(store, similarity_threshold=0.85)
        unrelated = make_node("n2", NodeType.DEAD_END, "Django ORM N+1 query in user list endpoint")
        result = dedup.deduplicate([unrelated], "project-ui", "session-2")
        assert len(result.stored) == 1
        assert len(result.dropped) == 0

    def test_dedup_reduces_redundancy_across_sessions(self, store):
        """Two sessions hitting the same bug should deduplicate to ~1 node."""
        session1_nodes = [
            make_node("n1", NodeType.SOLUTION,
                      "Fixed missing CORS header for the auth API endpoint in production config",
                      confidence=0.85),
        ]
        session2_nodes = [
            make_node("n1", NodeType.SOLUTION,
                      "Fixed missing CORS header for the auth API endpoint in the production configuration",
                      confidence=0.75),
        ]

        dag1 = make_dag("session-cors-1", session1_nodes)
        store.store_dag(dag1, "project-cors")
        before = store.node_count("project-cors")

        dedup = SemanticDeduplicator(store, similarity_threshold=0.70)
        result = dedup.deduplicate(session2_nodes, "project-cors", "session-cors-2")

        # The second one should be dropped (lower confidence)
        assert len(result.dropped) == 1
        # Node count should stay at 1
        assert store.node_count("project-cors") == before
