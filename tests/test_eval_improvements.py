"""Tests for the five evaluation improvements."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.schemas.reasoning import NodeType, ReasoningDAG, ReasoningNode, ReasoningEdge
from src.schemas.session import MessageRole, NormalizedSession, SessionMessage


def _session(n_messages: int = 20) -> NormalizedSession:
    now = datetime(2026, 1, 1, 12, 0, 0)
    return NormalizedSession(
        session_id="test-session",
        source_platform="claude_code",
        started_at=now,
        messages=[
            SessionMessage(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"message {i}",
            )
            for i in range(n_messages)
        ],
    )


def _dag(dead_end_at: list[int] = None, solution_at: list[int] = None, pivot_at: list[int] = None) -> ReasoningDAG:
    nodes = []
    for pos in (dead_end_at or []):
        nodes.append(ReasoningNode(
            node_id=f"de-{pos}", node_type=NodeType.DEAD_END,
            summary=f"dead end at {pos}", evidence="...",
            message_range=(pos, pos + 1), confidence=0.7,
        ))
    for pos in (solution_at or []):
        nodes.append(ReasoningNode(
            node_id=f"sol-{pos}", node_type=NodeType.SOLUTION,
            summary=f"solution at {pos}", evidence="...",
            message_range=(pos, pos + 1), confidence=0.8,
        ))
    for pos in (pivot_at or []):
        nodes.append(ReasoningNode(
            node_id=f"piv-{pos}", node_type=NodeType.PIVOT,
            summary=f"pivot at {pos}", evidence="...",
            message_range=(pos, pos + 1), confidence=0.6,
        ))
    return ReasoningDAG(session_id="test-session", nodes=nodes, edges=[])


# ═══════════════════════════════════════════════════════════════════
# Improvement #1: Real message index positions
# ═══════════════════════════════════════════════════════════════════


def test_retrieval_positions_uses_real_index():
    """When invocations have estimated_message_index, use them directly."""
    from src.evaluation.analyzer import SessionAnalyzer

    invocations = [
        {"skill": "search-memory", "estimated_message_index": 5},
        {"skill": "search-memory", "estimated_message_index": 15},
    ]
    positions = SessionAnalyzer._get_retrieval_positions(invocations, {"search-memory"}, 20)
    assert positions == [5, 15]


def test_retrieval_positions_falls_back_for_legacy():
    """When estimated_message_index is -1, fall back to even distribution."""
    from src.evaluation.analyzer import SessionAnalyzer

    invocations = [
        {"skill": "search-memory", "estimated_message_index": -1},
        {"skill": "search-memory", "estimated_message_index": -1},
    ]
    positions = SessionAnalyzer._get_retrieval_positions(invocations, {"search-memory"}, 30)
    assert len(positions) == 2
    # Even distribution of 2 invocations across 30 messages → ~10 and ~20
    assert positions[0] > 0 and positions[1] > positions[0]


def test_retrieval_positions_mixed_real_and_legacy():
    """Mix of real and legacy positions."""
    from src.evaluation.analyzer import SessionAnalyzer

    invocations = [
        {"skill": "search-memory", "estimated_message_index": 7},
        {"skill": "search-memory", "estimated_message_index": -1},
    ]
    positions = SessionAnalyzer._get_retrieval_positions(invocations, {"search-memory"}, 20)
    assert 7 in positions
    assert len(positions) == 2


def test_signal_a_uses_real_positions(tmp_path):
    """Signal A uses real message index to find errors resolved with memory."""
    from src.evaluation.analyzer import SessionAnalyzer
    from src.evaluation.logger import InteractionLogger

    logger = InteractionLogger(db_path=str(tmp_path / "eval.db"))
    analyzer = SessionAnalyzer(logger)

    session = _session(20)
    # Dead end at 10, solution at 14, retrieval at 11
    dag = _dag(dead_end_at=[10], solution_at=[14])

    invocations = [
        {"skill": "search-memory", "estimated_message_index": 11},
    ]
    count = analyzer._count_errors_resolved_with_memory(dag, invocations, 20)
    assert count == 1


def test_signal_c_uses_real_positions(tmp_path):
    """Signal C uses real message index to detect pivots after retrieval."""
    from src.evaluation.analyzer import SessionAnalyzer
    from src.evaluation.logger import InteractionLogger

    logger = InteractionLogger(db_path=str(tmp_path / "eval.db"))
    analyzer = SessionAnalyzer(logger)

    dag = _dag(pivot_at=[13])
    invocations = [
        {"skill": "search-memory", "estimated_message_index": 10},
    ]
    count = analyzer._count_pivots_after_retrieval(dag, invocations)
    # Pivot at 13, retrieval at 10, diff = 3 which is <= _PIVOT_WINDOW(5)
    assert count == 1


# ═══════════════════════════════════════════════════════════════════
# Improvement #2: Embedding-based pitfall matching
# ═══════════════════════════════════════════════════════════════════


def test_pitfall_avoidance_uses_embeddings(tmp_path):
    """Signal B uses embedding similarity instead of word overlap."""
    from src.evaluation.analyzer import SessionAnalyzer
    from src.evaluation.logger import InteractionLogger
    from src.schemas.memory import CognitiveProfile, Pitfall

    # Mock the store to return a profile with pitfalls
    profile = CognitiveProfile(
        project_id="proj1",
        last_updated=datetime(2026, 1, 1),
        pitfalls=[
            Pitfall(description="Database migrations fail when models aren't registered"),
            Pitfall(description="API rate limiting causes timeout errors"),
        ],
    )

    # Dead end that semantically matches the first pitfall
    dag = _dag(dead_end_at=[5])
    dag.nodes[0] = ReasoningNode(
        node_id="de-5", node_type=NodeType.DEAD_END,
        summary="The migration failed because the model was not registered in __init__",
        evidence="...", message_range=(5, 6), confidence=0.7,
    )

    # Invocations: pitfalls was called
    invocations = [{"skill": "pitfalls"}]

    # We need to mock the store
    def fake_embed(texts):
        # Return embeddings where pitfall[0] is close to the dead end,
        # and pitfall[1] is far from it
        embs = []
        for t in texts:
            if "migration" in t.lower() or "model" in t.lower():
                embs.append([0.9, 0.1, 0.0, 0.0])  # cluster A
            else:
                embs.append([0.0, 0.0, 0.9, 0.1])  # cluster B
        return embs

    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile
    mock_store.embed = fake_embed

    logger = InteractionLogger(db_path=str(tmp_path / "eval.db"))
    analyzer = SessionAnalyzer(logger, store_path=str(tmp_path / "store"))

    with patch("src.store.vector_store.MemoryStore", return_value=mock_store):
        surfaced, avoided = analyzer._count_pitfall_avoidance(dag, invocations, "proj1")

    # First pitfall matches the dead end (both in cluster A) → NOT avoided
    # Second pitfall doesn't match → avoided
    assert surfaced == 2
    assert avoided == 1


# ═══════════════════════════════════════════════════════════════════
# Improvement #4: Harmful memory (false positive) tracker
# ═══════════════════════════════════════════════════════════════════


def test_harmful_memory_detects_misleading_insight(tmp_path):
    """Signal D: loaded memory matches a subsequent dead end."""
    from src.evaluation.analyzer import SessionAnalyzer
    from src.evaluation.logger import InteractionLogger
    from src.schemas.memory import CognitiveProfile, Pitfall

    profile = CognitiveProfile(
        project_id="proj1",
        last_updated=datetime(2026, 1, 1),
        pitfalls=[
            Pitfall(description="The cache layer causes stale reads after writes"),
        ],
    )

    # Dead end that matches the pitfall — the memory may have led the agent
    # to investigate the cache when the real issue was elsewhere
    dag = _dag(dead_end_at=[8])
    dag.nodes[0] = ReasoningNode(
        node_id="de-8", node_type=NodeType.DEAD_END,
        summary="Investigated cache staleness but it was not the problem",
        evidence="...", message_range=(8, 9), confidence=0.7,
    )

    invocations = [{"skill": "cognitive-profile", "node_ids": "[]"}]

    def fake_embed(texts):
        embs = []
        for t in texts:
            if "cache" in t.lower() or "stale" in t.lower():
                embs.append([0.9, 0.1, 0.0, 0.0])
            else:
                embs.append([0.0, 0.0, 0.9, 0.1])
        return embs

    mock_store = MagicMock()
    mock_store.get_profile.return_value = profile
    mock_store.embed = fake_embed
    mock_store.nodes_col = MagicMock()
    mock_store.nodes_col.get.return_value = {"documents": []}

    logger = InteractionLogger(db_path=str(tmp_path / "eval.db"))
    analyzer = SessionAnalyzer(logger, store_path=str(tmp_path / "store"))

    with patch("src.store.vector_store.MemoryStore", return_value=mock_store):
        count = analyzer._count_harmful_memory(dag, invocations, "proj1")

    assert count == 1


def test_harmful_memory_zero_when_no_dead_ends(tmp_path):
    from src.evaluation.analyzer import SessionAnalyzer
    from src.evaluation.logger import InteractionLogger

    logger = InteractionLogger(db_path=str(tmp_path / "eval.db"))
    analyzer = SessionAnalyzer(logger)

    dag = _dag()  # no dead ends
    invocations = [{"skill": "cognitive-profile", "node_ids": "[]"}]
    count = analyzer._count_harmful_memory(dag, invocations, "proj1")
    assert count == 0


# ═══════════════════════════════════════════════════════════════════
# Improvement #5: Profile quality metrics
# ═══════════════════════════════════════════════════════════════════


def test_staleness_detects_missing_files(tmp_path):
    from src.evaluation.profile_quality import check_staleness
    from src.schemas.memory import CognitiveProfile, ArchitecturalInsight

    # Create a project dir with one file that exists
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "src").mkdir()
    (project_dir / "src" / "main.py").write_text("# exists")

    profile = CognitiveProfile(
        project_id="test",
        last_updated=datetime(2026, 1, 1),
        architectural_insights=[
            ArchitecturalInsight(
                component="backend",
                insight="The main entry point is src/main.py and utils/helpers.py handles parsing",
                confidence=0.8,
            ),
        ],
    )

    result = check_staleness(profile, project_dir)
    assert result["existing"] >= 1  # src/main.py found
    assert result["missing"] >= 1   # utils/helpers.py doesn't exist
    assert result["staleness_ratio"] > 0


def test_redundancy_detects_duplicates():
    from src.evaluation.profile_quality import check_redundancy
    from src.schemas.memory import CognitiveProfile, Pitfall

    profile = CognitiveProfile(
        project_id="test",
        last_updated=datetime(2026, 1, 1),
        pitfalls=[
            Pitfall(description="Database migrations fail when models aren't registered"),
            Pitfall(description="Database migrations fail because models aren't registered in __init__"),
            Pitfall(description="API rate limiting causes timeout errors on batch operations"),
        ],
    )

    # Use embedding function that clusters similar texts
    def fake_embed(texts):
        embs = []
        for t in texts:
            if "migration" in t.lower():
                embs.append([0.9, 0.1, 0.0, 0.0])
            else:
                embs.append([0.0, 0.0, 0.9, 0.1])
        return embs

    result = check_redundancy(profile, embed_fn=fake_embed, threshold=0.85)
    assert len(result["redundant_pairs"]) >= 1
    assert result["redundancy_ratio"] > 0


def test_redundancy_no_api_fallback():
    """Without embed_fn, falls back to word overlap."""
    from src.evaluation.profile_quality import check_redundancy
    from src.schemas.memory import CognitiveProfile, Pitfall

    profile = CognitiveProfile(
        project_id="test",
        last_updated=datetime(2026, 1, 1),
        pitfalls=[
            Pitfall(description="the quick brown fox"),
            Pitfall(description="a completely different sentence about elephants"),
        ],
    )

    result = check_redundancy(profile, embed_fn=None, threshold=0.5)
    # These two have no word overlap, so no redundancy
    assert len(result["redundant_pairs"]) == 0


def test_coverage_ratio():
    from src.evaluation.profile_quality import check_coverage
    from src.schemas.memory import CognitiveProfile

    profile = CognitiveProfile(
        project_id="test",
        last_updated=datetime(2026, 1, 1),
        session_count=3,
    )

    result = check_coverage(profile, total_sessions=10)
    assert result["coverage_ratio"] == 0.3


def test_run_quality_checks_combined(tmp_path):
    from src.evaluation.profile_quality import run_quality_checks
    from src.schemas.memory import CognitiveProfile

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    profile = CognitiveProfile(
        project_id="test",
        last_updated=datetime(2026, 1, 1),
        session_count=5,
    )

    result = run_quality_checks(profile, project_dir, total_sessions=10)
    assert "staleness" in result
    assert "redundancy" in result
    assert "coverage" in result
    assert result["coverage"]["coverage_ratio"] == 0.5


# ═══════════════════════════════════════════════════════════════════
# Logger schema: estimated_message_index column
# ═══════════════════════════════════════════════════════════════════


def test_logger_stores_message_index(tmp_path):
    from src.evaluation.logger import InteractionLogger

    logger = InteractionLogger(db_path=str(tmp_path / "eval.db"))
    logger.log_invocation(
        skill="search-memory",
        project_id="proj1",
        estimated_message_index=42,
    )

    invocations = logger.get_invocations(project_id="proj1")
    assert len(invocations) == 1
    assert invocations[0]["estimated_message_index"] == 42


def test_logger_default_message_index_is_minus_one(tmp_path):
    from src.evaluation.logger import InteractionLogger

    logger = InteractionLogger(db_path=str(tmp_path / "eval.db"))
    logger.log_invocation(skill="pitfalls", project_id="proj1")

    invocations = logger.get_invocations(project_id="proj1")
    assert invocations[0]["estimated_message_index"] == -1
