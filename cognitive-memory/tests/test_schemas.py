"""Schema validation tests — Phase 0 exit criteria."""
import pytest
from datetime import datetime, timezone
from src.schemas import (
    MessageRole,
    SessionMessage,
    NormalizedSession,
    NodeType,
    ReasoningNode,
    ReasoningEdge,
    ReasoningDAG,
    Pitfall,
    DiagnosticStrategy,
    ArchitecturalInsight,
    CognitiveProfile,
)


def utcnow():
    return datetime.now(timezone.utc)


# ── SessionMessage ──────────────────────────────────────────────────────────

def test_session_message_minimal():
    msg = SessionMessage(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.timestamp is None
    assert msg.files_referenced == []
    assert msg.files_modified == []


def test_session_message_tool_call():
    msg = SessionMessage(
        role=MessageRole.TOOL_CALL,
        content='{"file_path": "/foo/bar.py"}',
        tool_name="Read",
        files_referenced=["/foo/bar.py"],
    )
    assert msg.tool_name == "Read"
    assert "/foo/bar.py" in msg.files_referenced


# ── NormalizedSession ───────────────────────────────────────────────────────

def test_normalized_session():
    session = NormalizedSession(
        session_id="test-session-001",
        source_platform="claude_code",
        project_path="/Users/test/project",
        started_at=utcnow(),
        messages=[
            SessionMessage(role=MessageRole.USER, content="Fix the bug"),
            SessionMessage(role=MessageRole.ASSISTANT, content="I'll look into it"),
        ],
    )
    assert len(session.messages) == 2
    assert session.source_platform == "claude_code"
    assert session.raw_metadata == {}


def test_normalized_session_json_roundtrip():
    session = NormalizedSession(
        session_id="roundtrip-test",
        source_platform="claude_code",
        started_at=utcnow(),
        messages=[SessionMessage(role=MessageRole.USER, content="test")],
    )
    serialized = session.model_dump_json()
    restored = NormalizedSession.model_validate_json(serialized)
    assert restored.session_id == session.session_id
    assert len(restored.messages) == 1


# ── ReasoningNode / ReasoningEdge / ReasoningDAG ───────────────────────────

def test_reasoning_node():
    node = ReasoningNode(
        node_id="node-1",
        node_type=NodeType.HYPOTHESIS,
        summary="Agent forms a hypothesis about the bug",
        evidence="User reports 401 errors",
        message_range=(0, 3),
        confidence=0.85,
    )
    assert node.node_type == NodeType.HYPOTHESIS
    assert 0.0 <= node.confidence <= 1.0


def test_reasoning_dag():
    nodes = [
        ReasoningNode(
            node_id="n1",
            node_type=NodeType.INVESTIGATION,
            summary="Read auth middleware",
            evidence="Read tool call on auth.py",
            message_range=(0, 2),
        ),
        ReasoningNode(
            node_id="n2",
            node_type=NodeType.PIVOT,
            summary="Found key mismatch, changed approach",
            evidence="OLD_SECRET_KEY vs SECRET_KEY",
            message_range=(3, 6),
        ),
        ReasoningNode(
            node_id="n3",
            node_type=NodeType.SOLUTION,
            summary="Fixed login.py to use correct key",
            evidence="Edit tool call",
            message_range=(7, 9),
        ),
    ]
    edges = [
        ReasoningEdge(source_id="n1", target_id="n2", relationship="led_to"),
        ReasoningEdge(source_id="n2", target_id="n3", relationship="led_to"),
    ]
    dag = ReasoningDAG(
        session_id="test-session-001",
        nodes=nodes,
        edges=edges,
        pivot_nodes=["n2"],
        noise_ratio=0.75,
    )
    assert len(dag.nodes) == 3
    assert "n2" in dag.pivot_nodes
    assert dag.noise_ratio == 0.75


# ── CognitiveProfile ────────────────────────────────────────────────────────

def test_cognitive_profile():
    profile = CognitiveProfile(
        project_id="my-project",
        last_updated=utcnow(),
        architectural_insights=[
            ArchitecturalInsight(
                component="auth",
                insight="JWT signing and verification must use the same SECRET_KEY",
                confidence=0.9,
            )
        ],
        pitfalls=[
            Pitfall(
                description="Using different variable names for the same secret key across modules",
                frequency=2,
                severity="high",
                resolution_strategy="Centralize secret key access in a single config module",
            )
        ],
        diagnostic_strategies=[
            DiagnosticStrategy(
                trigger="401 authentication errors",
                steps=[
                    "Check token signing key in login module",
                    "Check token verification key in auth middleware",
                    "Verify both reference the same env variable",
                ],
                success_rate=1.0,
                source_sessions=["synthetic-debug-01"],
            )
        ],
        key_patterns=["JWT token auth via HS256"],
        anti_patterns=["Hardcoded fallback secret keys"],
        session_count=1,
    )
    assert len(profile.pitfalls) == 1
    assert profile.pitfalls[0].severity == "high"
    assert profile.session_count == 1

    # JSON roundtrip
    data = profile.model_dump_json()
    restored = CognitiveProfile.model_validate_json(data)
    assert restored.project_id == "my-project"
    assert len(restored.diagnostic_strategies) == 1
