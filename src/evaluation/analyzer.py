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

import numpy as np

from ..schemas.reasoning import NodeType, ReasoningDAG
from ..schemas.session import NormalizedSession
from .logger import InteractionLogger


# How many messages after a retrieval to look for a resolution
_RESOLUTION_WINDOW = 8
# How many messages after a retrieval to look for a pivot
_PIVOT_WINDOW = 5
# Cosine similarity threshold for pitfall-to-dead-end matching (improvement #2)
_PITFALL_EMBED_THRESHOLD = 0.70
# Cosine similarity threshold for harmful-memory detection (improvement #4)
_HARMFUL_MEMORY_THRESHOLD = 0.70


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


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

        # Signal B: Pitfall avoidance (embedding-based matching)
        pitfalls_surfaced, pitfalls_avoided = self._count_pitfall_avoidance(
            dag, invocations, project_id
        )

        # Signal C: Pivots after retrieval
        pivots_after = self._count_pivots_after_retrieval(dag, invocations)

        # Signal D: Harmful memory — memory loaded then a matching dead end
        harmful_count = self._count_harmful_memory(dag, invocations, project_id)

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
            "harmful_memory_count": harmful_count,
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

    # ── Retrieval position helper ─────────────────────────────────

    @staticmethod
    def _get_retrieval_positions(
        invocations: list[dict],
        skills: set[str],
        total_messages: int,
    ) -> list[int]:
        """Extract message positions for invocations of the given skills.

        Uses the real `estimated_message_index` recorded at invocation
        time (improvement #1). Falls back to even-distribution estimation
        only for legacy invocations that don't have the field.
        """
        relevant = [inv for inv in invocations if inv.get("skill") in skills]
        if not relevant:
            return []

        positions: list[int] = []
        needs_fallback = []
        for i, inv in enumerate(relevant):
            idx = inv.get("estimated_message_index", -1)
            if isinstance(idx, int) and idx >= 0:
                positions.append(idx)
            else:
                needs_fallback.append(i)

        # Fallback for legacy invocations: distribute evenly
        if needs_fallback and total_messages > 0:
            for i in needs_fallback:
                estimated_pos = int(
                    (i + 1) / (len(relevant) + 1) * total_messages
                )
                positions.append(estimated_pos)

        return sorted(positions)

    # ── Signal A: Error resolved using memory ─────────────────────

    def _count_errors_resolved_with_memory(
        self,
        dag: ReasoningDAG,
        invocations: list[dict],
        total_messages: int,
    ) -> int:
        """Count errors where memory retrieval preceded the resolution.

        Pattern: DEAD_END at msg X → retrieval near X → SOLUTION within X+WINDOW
        """
        retrieval_positions = self._get_retrieval_positions(
            invocations, {"search-memory", "diagnose"}, total_messages
        )
        if not retrieval_positions:
            return 0

        count = 0
        for dead_end in [n for n in dag.nodes if n.node_type == NodeType.DEAD_END]:
            de_pos = dead_end.message_range[0]
            for rp in retrieval_positions:
                if abs(rp - de_pos) <= _RESOLUTION_WINDOW:
                    for solution in [n for n in dag.nodes if n.node_type == NodeType.SOLUTION]:
                        if (solution.message_range[0] > de_pos and
                                solution.message_range[0] <= de_pos + _RESOLUTION_WINDOW):
                            count += 1
                            break
                    break

        return count

    # ── Signal B: Pitfall avoidance (embedding similarity) ────────

    def _count_pitfall_avoidance(
        self,
        dag: ReasoningDAG,
        invocations: list[dict],
        project_id: str,
    ) -> tuple[int, int]:
        """Count pitfalls surfaced and how many were avoided.

        Uses embedding cosine similarity (threshold 0.70) instead of
        word overlap to match pitfalls against dead ends.
        """
        pitfall_invocations = [
            inv for inv in invocations
            if inv.get("skill") in ("pitfalls", "cognitive-profile")
        ]
        if not pitfall_invocations:
            return 0, 0

        pitfalls_surfaced = 0
        pitfalls_avoided = 0

        try:
            if self._store_path:
                from src.store.vector_store import MemoryStore
                store = MemoryStore(persist_dir=self._store_path)
                profile = store.get_profile(project_id)
                if profile and profile.pitfalls:
                    pitfalls_surfaced = len(profile.pitfalls)

                    dead_end_summaries = [
                        n.summary
                        for n in dag.nodes
                        if n.node_type == NodeType.DEAD_END
                    ]

                    if not dead_end_summaries:
                        # No dead ends → every pitfall was avoided
                        return pitfalls_surfaced, pitfalls_surfaced

                    # Embed pitfall descriptions and dead end summaries
                    pitfall_texts = [p.description for p in profile.pitfalls]
                    all_texts = pitfall_texts + dead_end_summaries
                    all_embs = store.embed(all_texts)

                    pitfall_embs = all_embs[:len(pitfall_texts)]
                    de_embs = all_embs[len(pitfall_texts):]

                    for p_emb in pitfall_embs:
                        matched = False
                        for de_emb in de_embs:
                            sim = _cosine_sim(p_emb, de_emb)
                            if sim >= _PITFALL_EMBED_THRESHOLD:
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
        total_msgs = max(n.message_range[1] for n in dag.nodes) if dag.nodes else 0
        retrieval_positions = self._get_retrieval_positions(
            invocations, {"search-memory", "diagnose"}, total_msgs
        )
        if not retrieval_positions:
            return 0

        count = 0
        for pivot in [n for n in dag.nodes if n.node_type == NodeType.PIVOT]:
            pivot_pos = pivot.message_range[0]
            for rp in retrieval_positions:
                if 0 < pivot_pos - rp <= _PIVOT_WINDOW:
                    count += 1
                    break

        return count

    # ── Signal D: Harmful memory (false positive) ─────────────────

    def _count_harmful_memory(
        self,
        dag: ReasoningDAG,
        invocations: list[dict],
        project_id: str,
    ) -> int:
        """Count cases where loaded memory may have actively misled the agent.

        Pattern: the agent loaded pitfalls/insights via /cognitive-profile
        or /pitfalls, and then hit a dead end that semantically matches
        that memory. The memory may have pointed the agent in a wrong direction.
        """
        memory_invocations = [
            inv for inv in invocations
            if inv.get("skill") in ("cognitive-profile", "pitfalls", "search-memory", "diagnose")
        ]
        if not memory_invocations:
            return 0

        dead_end_summaries = [
            n.summary for n in dag.nodes
            if n.node_type == NodeType.DEAD_END
        ]
        if not dead_end_summaries:
            return 0

        count = 0
        try:
            if self._store_path:
                from src.store.vector_store import MemoryStore
                store = MemoryStore(persist_dir=self._store_path)

                # Collect all memory texts that were surfaced
                memory_texts: list[str] = []
                profile = store.get_profile(project_id)
                if profile:
                    memory_texts.extend(p.description for p in profile.pitfalls)
                    memory_texts.extend(i.insight for i in profile.architectural_insights)

                # Also include search result texts from retrieval invocations
                for inv in memory_invocations:
                    node_ids_json = inv.get("node_ids", "[]")
                    try:
                        nids = json.loads(node_ids_json) if node_ids_json else []
                    except Exception:
                        nids = []
                    if nids:
                        try:
                            fetched = store.nodes_col.get(ids=nids, include=["documents"])
                            memory_texts.extend(fetched.get("documents", []))
                        except Exception:
                            pass

                if not memory_texts:
                    return 0

                # Embed and compare
                all_texts = memory_texts + dead_end_summaries
                all_embs = store.embed(all_texts)

                mem_embs = all_embs[:len(memory_texts)]
                de_embs = all_embs[len(memory_texts):]

                # For each dead end, check if any loaded memory matches it
                matched_de = set()
                for j, de_emb in enumerate(de_embs):
                    for mem_emb in mem_embs:
                        sim = _cosine_sim(mem_emb, de_emb)
                        if sim >= _HARMFUL_MEMORY_THRESHOLD:
                            matched_de.add(j)
                            break

                count = len(matched_de)
        except Exception:
            pass

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
