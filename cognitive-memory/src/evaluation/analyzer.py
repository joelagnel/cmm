"""Post-session analyzer — determines whether retrieved memories were helpful.

Runs after a session ends. Reads the session transcript and the interaction
log, then correlates them to measure helpfulness via three signals:

Signal A — Error resolved using memory:
    A DEAD_END node is followed by a memory retrieval, then a SOLUTION node
    within N messages. The memory helped resolve the error.

Signal B — Pitfall avoidance:
    A pitfall was surfaced via /pitfalls or /cognitive-profile. If no DEAD_END
    node in the session matches that pitfall (semantic similarity), it was avoided.

Signal C — Solution efficiency:
    Count dead ends, pivots, messages to first solution. Compare against
    sessions without memory for the same project.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..schemas.reasoning import NodeType, ReasoningDAG
from ..schemas.session import NormalizedSession
from .logger import InteractionLogger


# How many messages after a retrieval to look for a resolution
_RESOLUTION_WINDOW = 8
# How many messages after a retrieval to look for a pivot
_PIVOT_WINDOW = 5
# Similarity threshold for matching pitfalls to dead ends
_PITFALL_MATCH_THRESHOLD = 0.6


class SessionAnalyzer:
    """Analyze a completed session against its interaction log."""

    def __init__(
        self,
        logger: InteractionLogger,
        store_path: str | None = None,
    ):
        self.logger = logger
        self._store_path = store_path

    def analyze(
        self,
        session: NormalizedSession,
        dag: ReasoningDAG,
        project_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a session's interaction with cognitive memory.

        Args:
            session: The parsed session transcript
            dag: The reasoning DAG extracted from the session
            project_id: Project identifier
            session_id: Override session ID (default: from DAG)

        Returns:
            Evaluation dict ready for logger.save_session_evaluation()
        """
        sid = session_id or dag.session_id

        # Get all invocations for this session
        invocations = self.logger.get_invocations(session_id=sid)

        # Basic session stats
        total_messages = len(session.messages)
        duration = 0.0
        if session.started_at and session.ended_at:
            duration = (session.ended_at - session.started_at).total_seconds()
        elif total_messages >= 2 and session.messages[0].timestamp and session.messages[-1].timestamp:
            duration = (session.messages[-1].timestamp - session.messages[0].timestamp).total_seconds()

        # DAG stats
        dead_ends = [n for n in dag.nodes if n.node_type == NodeType.DEAD_END]
        pivots = [n for n in dag.nodes if n.node_type == NodeType.PIVOT]
        solutions = [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]

        msgs_to_first_solution = 0
        if solutions:
            msgs_to_first_solution = min(s.message_range[0] for s in solutions)

        # Signal A: Errors resolved with memory
        errors_resolved = self._count_errors_resolved_with_memory(
            dag, invocations, total_messages
        )

        # Signal B: Pitfall avoidance
        pitfalls_surfaced, pitfalls_avoided = self._count_pitfall_avoidance(
            dag, invocations, project_id
        )

        # Signal C: Pivots after retrieval
        pivots_after = self._count_pivots_after_retrieval(dag, invocations)

        # Was memory used at session start? (first invocation in first 3 messages)
        memory_at_start = 0
        if invocations:
            early_skills = {"cognitive-profile", "pitfalls"}
            for inv in invocations:
                if inv.get("skill") in early_skills:
                    memory_at_start = 1
                    break

        evaluation = {
            "session_id": sid,
            "project_id": project_id,
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_invocations": len(invocations),
            "memory_used_at_start": memory_at_start,
            "errors_encountered": len(dead_ends),
            "errors_resolved_with_memory": errors_resolved,
            "pitfalls_surfaced": pitfalls_surfaced,
            "pitfalls_avoided": pitfalls_avoided,
            "pivots_after_retrieval": pivots_after,
            "total_dead_ends": len(dead_ends),
            "total_pivots": len(pivots),
            "messages_to_first_solution": msgs_to_first_solution,
            "total_messages": total_messages,
            "total_nodes": len(dag.nodes),
            "duration_seconds": duration,
        }

        # Persist
        self.logger.save_session_evaluation(evaluation)

        return evaluation

    # ── Signal A: Error resolved using memory ─────────────────────

    def _count_errors_resolved_with_memory(
        self,
        dag: ReasoningDAG,
        invocations: list[dict],
        total_messages: int,
    ) -> int:
        """Count errors where memory retrieval preceded the resolution.

        Pattern: DEAD_END at msg X → retrieval → SOLUTION within X+WINDOW
        """
        if not invocations:
            return 0

        # Build a set of message positions where retrievals happened
        # We estimate position from invocation order within the session
        retrieval_positions = set()
        search_invocations = [
            inv for inv in invocations
            if inv.get("skill") in ("search-memory", "diagnose")
        ]

        # Map invocation timestamps to approximate message positions
        for i, inv in enumerate(search_invocations):
            # Rough estimate: distribute invocations across the session
            if total_messages > 0 and len(search_invocations) > 0:
                estimated_pos = int(
                    (i + 1) / (len(search_invocations) + 1) * total_messages
                )
                retrieval_positions.add(estimated_pos)

        count = 0
        for dead_end in [n for n in dag.nodes if n.node_type == NodeType.DEAD_END]:
            de_pos = dead_end.message_range[0]
            # Was there a retrieval near this dead end?
            for rp in retrieval_positions:
                if abs(rp - de_pos) <= _RESOLUTION_WINDOW:
                    # Is there a solution after the dead end within the window?
                    for solution in [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]:
                        if (solution.message_range[0] > de_pos and
                                solution.message_range[0] <= de_pos + _RESOLUTION_WINDOW):
                            count += 1
                            break
                    break

        return count

    # ── Signal B: Pitfall avoidance ───────────────────────────────

    def _count_pitfall_avoidance(
        self,
        dag: ReasoningDAG,
        invocations: list[dict],
        project_id: str,
    ) -> tuple[int, int]:
        """Count pitfalls surfaced and how many were avoided.

        A pitfall is "avoided" if it was surfaced via /pitfalls or
        /cognitive-profile, and no DEAD_END node in the session
        semantically matches that pitfall.
        """
        # Check if pitfalls or cognitive-profile was called
        pitfall_invocations = [
            inv for inv in invocations
            if inv.get("skill") in ("pitfalls", "cognitive-profile")
        ]
        if not pitfall_invocations:
            return 0, 0

        # Load the profile to get actual pitfall descriptions
        pitfalls_surfaced = 0
        pitfalls_avoided = 0

        try:
            if self._store_path:
                from src.store.vector_store import MemoryStore
                store = MemoryStore(persist_dir=self._store_path)
                profile = store.get_profile(project_id)
                if profile and profile.pitfalls:
                    pitfalls_surfaced = len(profile.pitfalls)

                    # Get dead end summaries
                    dead_end_summaries = [
                        n.summary.lower()
                        for n in dag.nodes
                        if n.node_type == NodeType.DEAD_END
                    ]

                    # For each pitfall, check if a dead end matches
                    for pitfall in profile.pitfalls:
                        pitfall_words = set(pitfall.description.lower().split())
                        matched = False
                        for de_summary in dead_end_summaries:
                            de_words = set(de_summary.split())
                            overlap = len(pitfall_words & de_words)
                            if overlap >= 3:  # at least 3 words in common
                                matched = True
                                break
                        if not matched:
                            pitfalls_avoided += 1
        except Exception:
            pass

        return pitfalls_surfaced, pitfalls_avoided

    # ── Signal C: Pivots after retrieval ──────────────────────────

    def _count_pivots_after_retrieval(
        self,
        dag: ReasoningDAG,
        invocations: list[dict],
    ) -> int:
        """Count pivot nodes that occur shortly after a memory retrieval."""
        if not invocations:
            return 0

        search_invocations = [
            inv for inv in invocations
            if inv.get("skill") in ("search-memory", "diagnose")
        ]
        if not search_invocations:
            return 0

        # Estimate retrieval positions
        total_msgs = max(n.message_range[1] for n in dag.nodes) if dag.nodes else 0
        retrieval_positions = []
        for i, inv in enumerate(search_invocations):
            if total_msgs > 0:
                estimated_pos = int(
                    (i + 1) / (len(search_invocations) + 1) * total_msgs
                )
                retrieval_positions.append(estimated_pos)

        count = 0
        pivots = [n for n in dag.nodes if n.node_type == NodeType.PIVOT]
        for pivot in pivots:
            pivot_pos = pivot.message_range[0]
            for rp in retrieval_positions:
                if 0 < pivot_pos - rp <= _PIVOT_WINDOW:
                    count += 1
                    break

        return count

    # ── Compare two sessions ──────────────────────────────────────

    def compare_sessions(
        self,
        baseline_session: NormalizedSession,
        baseline_dag: ReasoningDAG,
        assisted_session: NormalizedSession,
        assisted_dag: ReasoningDAG,
        project_id: str,
    ) -> dict[str, Any]:
        """Compare a baseline (no memory) session with a memory-assisted session.

        Returns a comparison dict with both session stats side by side.
        """
        def _session_stats(session: NormalizedSession, dag: ReasoningDAG) -> dict:
            dead_ends = [n for n in dag.nodes if n.node_type == NodeType.DEAD_END]
            pivots = [n for n in dag.nodes if n.node_type == NodeType.PIVOT]
            solutions = [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]
            hypotheses = [n for n in dag.nodes if n.node_type == NodeType.HYPOTHESIS]

            msgs_to_solution = 0
            if solutions:
                msgs_to_solution = min(s.message_range[0] for s in solutions)

            duration = 0.0
            if len(session.messages) >= 2:
                first = session.messages[0].timestamp
                last = session.messages[-1].timestamp
                if first and last:
                    duration = (last - first).total_seconds()

            return {
                "total_messages": len(session.messages),
                "total_nodes": len(dag.nodes),
                "dead_ends": len(dead_ends),
                "pivots": len(pivots),
                "solutions": len(solutions),
                "hypotheses": len(hypotheses),
                "messages_to_first_solution": msgs_to_solution,
                "duration_seconds": duration,
            }

        baseline_stats = _session_stats(baseline_session, baseline_dag)
        assisted_stats = _session_stats(assisted_session, assisted_dag)

        # Get invocation data for assisted session
        assisted_invocations = self.logger.get_invocations(
            session_id=assisted_dag.session_id
        )

        # Calculate reductions
        def _reduction(before, after):
            if before == 0:
                return 0.0
            return round((before - after) / before * 100, 1)

        return {
            "project_id": project_id,
            "baseline": baseline_stats,
            "assisted": assisted_stats,
            "memory_invocations": len(assisted_invocations),
            "reductions": {
                "messages": _reduction(
                    baseline_stats["total_messages"],
                    assisted_stats["total_messages"],
                ),
                "dead_ends": _reduction(
                    baseline_stats["dead_ends"],
                    assisted_stats["dead_ends"],
                ),
                "pivots": _reduction(
                    baseline_stats["pivots"],
                    assisted_stats["pivots"],
                ),
                "messages_to_solution": _reduction(
                    baseline_stats["messages_to_first_solution"],
                    assisted_stats["messages_to_first_solution"],
                ),
                "duration": _reduction(
                    baseline_stats["duration_seconds"],
                    assisted_stats["duration_seconds"],
                ),
            },
        }
