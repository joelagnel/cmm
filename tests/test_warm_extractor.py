"""Warm-tier extractor tests — Phase 4.2 exit criteria."""
import time
import pytest
from pathlib import Path
from datetime import datetime

from src.extraction.warm_extractor import WarmExtractor
from src.ingestion import ClaudeCodeParser
from src.schemas.session import NormalizedSession, SessionMessage, MessageRole
from src.schemas.reasoning import NodeType


FIXTURES = Path(__file__).parent.parent / "fixtures"
SYNTHETIC = FIXTURES / "synthetic"


def _make_session(messages: list[tuple[str, str]]) -> NormalizedSession:
    """Create a NormalizedSession from (role, content) pairs."""
    msgs = []
    for role_str, content in messages:
        role = MessageRole(role_str)
        msgs.append(SessionMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
        ))
    return NormalizedSession(
        session_id="test-session",
        source_platform="test",
        started_at=datetime.now(),
        messages=msgs,
    )


def test_extract_produces_dag():
    """WarmExtractor produces a valid ReasoningDAG."""
    session = _make_session([
        ("user", "The tests are failing with import errors"),
        ("assistant", "I suspect the issue is with the module path configuration. Let me investigate."),
        ("assistant", "I found that the __init__.py file is missing from the package directory."),
        ("assistant", "I fixed the issue by creating the __init__.py file. All tests pass now."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    assert dag.session_id == "test-session"
    assert len(dag.nodes) > 0
    assert all(n.confidence > 0 for n in dag.nodes)
    assert all(n.confidence <= 1.0 for n in dag.nodes)


def test_extract_classifies_hypothesis():
    """Messages with hypothesis keywords are classified correctly."""
    session = _make_session([
        ("user", "Why is the server crashing?"),
        ("assistant", "I suspect the issue might be with the database connection pool. I believe the timeout is too short."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    hypothesis_nodes = [n for n in dag.nodes if n.node_type == NodeType.HYPOTHESIS]
    assert len(hypothesis_nodes) >= 1


def test_extract_classifies_discovery():
    """Messages with discovery keywords are classified correctly."""
    session = _make_session([
        ("user", "Check the config"),
        ("assistant", "I discovered that the config file has an unexpected setting that reveals the root cause of the issue."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    discovery_nodes = [n for n in dag.nodes if n.node_type == NodeType.DISCOVERY]
    assert len(discovery_nodes) >= 1


def test_extract_classifies_pivot():
    """Messages with pivot keywords are classified correctly."""
    session = _make_session([
        ("user", "Try fixing the auth"),
        ("assistant", "The approach doesn't work. I need to rethink this. Let me try a different approach instead of patching the middleware."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    pivot_nodes = [n for n in dag.nodes if n.node_type == NodeType.PIVOT]
    assert len(pivot_nodes) >= 1


def test_extract_classifies_solution():
    """Messages with solution keywords are classified correctly."""
    session = _make_session([
        ("user", "Fix the build"),
        ("assistant", "I fixed the build configuration. All tests pass and the build succeeds. The solution was to update the module paths."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    solution_nodes = [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]
    assert len(solution_nodes) >= 1


def test_extract_classifies_dead_end():
    """Messages indicating failure are classified as dead ends."""
    session = _make_session([
        ("user", "Try the cache approach"),
        ("assistant", "The caching approach completely failed and is not working at all. No luck with this path, it is a dead end."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    dead_ends = [n for n in dag.nodes if n.node_type == NodeType.DEAD_END]
    assert len(dead_ends) >= 1


def test_extract_finds_error_resolution():
    """Error-resolution pairs are detected."""
    session = _make_session([
        ("user", "Run the tests"),
        ("assistant", "Running tests now..."),
        ("tool_result", "ModuleNotFoundError: No module named 'mypackage'"),
        ("assistant", "I fixed the import error by installing the package. All tests pass now and the solution works."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    # Should find at least a solution node from the error-resolution pair
    solution_nodes = [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]
    assert len(solution_nodes) >= 1


def test_extract_finds_explicit_conclusions():
    """Explicit conclusion statements are captured."""
    session = _make_session([
        ("user", "What caused the bug?"),
        ("assistant", "After investigation, I found that the issue was caused by a race condition in the event loop."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    # The conclusion heuristic should produce a discovery node
    discoveries = [n for n in dag.nodes if n.node_type == NodeType.DISCOVERY]
    assert len(discoveries) >= 1


def test_extract_builds_edges():
    """Extracted DAG has edges connecting sequential nodes."""
    session = _make_session([
        ("user", "Debug the issue"),
        ("assistant", "I suspect the problem is in the auth module. Let me check."),
        ("assistant", "I found that the token validation was using the wrong key."),
        ("assistant", "I fixed the token validation. All tests pass now."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    assert len(dag.edges) == len(dag.nodes) - 1
    # Verify edges connect sequential nodes
    for i, edge in enumerate(dag.edges):
        assert edge.source_id == dag.nodes[i].node_id
        assert edge.target_id == dag.nodes[i + 1].node_id


def test_extract_marks_pivots():
    """Pivot and dead-end nodes are tracked in pivot_nodes.

    When dead_end/pivot keywords are strong enough and not consumed
    by an error-resolution pair, they appear in pivot_nodes.
    """
    session = _make_session([
        ("user", "Fix the performance issue in the API response times for the dashboard endpoint"),
        ("assistant", "I suspect the bottleneck is in the database query layer. Let me investigate the query execution plans and connection pool settings to understand the root cause."),
        ("user", "What did you find from checking the database queries and pool configuration?"),
        ("assistant", "The database query optimization approach completely failed and is not working at all. No luck with this dead end path. Nothing here helps."),
        ("user", "Ok try something else then"),
        ("assistant", "I need to rethink and pivot to a completely different approach instead of the database layer. On second thought the bottleneck is elsewhere entirely."),
    ])

    extractor = WarmExtractor(error_resolution_window=3)
    dag = extractor.extract(session)

    # Dead ends and pivots should be tracked
    pivot_or_dead = [n for n in dag.nodes if n.node_type in (NodeType.DEAD_END, NodeType.PIVOT)]
    assert len(pivot_or_dead) >= 1
    assert len(dag.pivot_nodes) >= 1


def test_extract_under_one_second_synthetic():
    """Warm extraction completes in under 1 second on synthetic sessions."""
    parser = ClaudeCodeParser()
    session = parser.parse_file(SYNTHETIC / "debugging_with_pivot.jsonl")

    extractor = WarmExtractor()
    start = time.perf_counter()
    dag = extractor.extract(session)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Warm extraction took {elapsed:.2f}s (must be <1s)"
    assert len(dag.nodes) > 0


def test_extract_skips_short_messages():
    """Messages shorter than min_message_length are skipped."""
    session = _make_session([
        ("user", "hi"),
        ("assistant", "ok"),
        ("assistant", "I suspect the database connection is timing out due to misconfigured pool settings."),
    ])

    extractor = WarmExtractor(min_message_length=40)
    dag = extractor.extract(session)

    # Only the long message should produce a node
    assert len(dag.nodes) == 1


def test_extract_noise_ratio():
    """Noise ratio is calculated correctly."""
    session = _make_session([
        ("user", "Hello"),
        ("assistant", "Hi there"),
        ("user", "Check the config"),
        ("assistant", "I suspect there might be a configuration issue with the database timeout settings."),
        ("user", "Fix it"),
        ("assistant", "I fixed the configuration. The timeout was set too low. All tests pass and the solution works now."),
    ])

    extractor = WarmExtractor()
    dag = extractor.extract(session)

    # Noise ratio = 1 - (nodes / messages)
    assert 0 < dag.noise_ratio < 1
    assert dag.noise_ratio == pytest.approx(1 - len(dag.nodes) / len(session.messages), abs=0.01)
