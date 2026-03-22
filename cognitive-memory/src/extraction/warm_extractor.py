"""Warm-tier heuristic extractor — fast extraction without LLM calls.

Runs in under 1 second. Uses pattern matching, keyword analysis, and
structural heuristics to classify reasoning steps. Produces lower-confidence
nodes than the full LLM extractor, but is fast enough for real-time ingestion.

Four heuristic strategies:
    1. Error-resolution pairs — errors followed by successful fixes
    2. File modification patterns — files changed together (coupling signal)
    3. Repeated attempts — same action tried multiple times (difficulty signal)
    4. Explicit conclusions — "I found that...", "The issue was...", etc.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.schemas.reasoning import (
    NodeType,
    ReasoningDAG,
    ReasoningEdge,
    ReasoningNode,
)
from src.schemas.session import MessageRole, NormalizedSession


# ── Keyword patterns ──────────────────────────────────────────────

_HYPOTHESIS_PATTERNS = [
    r"\b(?:hypothesis|suspect|theory|believe|think|assume|guess|likely|probably|might be)\b",
    r"\b(?:could be|looks like it might|my guess is|i think the issue)\b",
    r"\b(?:let me try|let's try|approach|strategy)\b",
]

_DISCOVERY_PATTERNS = [
    r"\b(?:found|discovered|notice|interesting|unexpected|reveals?|turns out)\b",
    r"\b(?:actually|the real issue|root cause|identified)\b",
    r"\b(?:aha|that explains|now i see|the problem (?:is|was))\b",
]

_PIVOT_PATTERNS = [
    r"\b(?:instead|different approach|pivot|change direction|wrong approach)\b",
    r"\b(?:doesn't work|won't work|need to rethink|let me reconsider)\b",
    r"\b(?:actually,? (?:let me|we should)|on second thought|better approach)\b",
    r"\b(?:scrap|abandon|back to|go back|revert)\b",
]

_SOLUTION_PATTERNS = [
    r"\b(?:fixed|resolved|solution|works? now|passing|success|completed)\b",
    r"\b(?:all tests pass|build succeeds|implemented|done)\b",
    r"\b(?:that (?:fixed|solved|resolved) it)\b",
]

_DEAD_END_PATTERNS = [
    r"\b(?:failed|didn't work|not working|no luck|error persists)\b",
    r"\b(?:still broken|same error|doesn't help|dead end)\b",
    r"\b(?:that (?:didn't|does not) (?:work|help|fix))\b",
]

_ERROR_PATTERNS = [
    r"(?:Error|Exception|Traceback|FAILED|error\[|ERR!|fatal:)",
    r"(?:ModuleNotFoundError|ImportError|TypeError|ValueError|KeyError)",
    r"(?:SyntaxError|IndentationError|NameError|AttributeError)",
    r"(?:ConnectionError|TimeoutError|FileNotFoundError|PermissionError)",
    r"(?:ENOENT|EACCES|ECONNREFUSED|segfault|panic|assertion failed)",
]

_CONCLUSION_PATTERNS = [
    r"(?:i found that|the issue was|the problem was|the fix is|the solution is)",
    r"(?:this (?:approach|method|pattern) works|key insight|lesson learned)",
    r"(?:important to note|for future reference|remember that)",
]


def _match_score(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in the text."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower, re.IGNORECASE))


# ── Heuristic strategies ─────────────────────────────────────────

@dataclass
class _QuickNode:
    """Internal intermediate node before final ReasoningNode construction."""
    node_type: NodeType
    summary: str
    evidence: str
    msg_start: int
    msg_end: int
    confidence: float
    source: str  # which heuristic generated this


class WarmExtractor:
    """Fast heuristic extraction for between-session processing.

    Produces a ReasoningDAG with lower-confidence nodes (0.3–0.7)
    compared to LLM extraction (0.5–0.95). Designed to run in <1s.
    """

    def __init__(
        self,
        min_message_length: int = 40,
        error_resolution_window: int = 8,
    ):
        self.min_message_length = min_message_length
        self.error_resolution_window = error_resolution_window

    def extract(self, session: NormalizedSession) -> ReasoningDAG:
        """Run all heuristics and merge into a single DAG."""
        quick_nodes: list[_QuickNode] = []

        quick_nodes.extend(self._classify_messages(session))
        quick_nodes.extend(self._find_error_resolutions(session))
        quick_nodes.extend(self._find_explicit_conclusions(session))

        # Deduplicate overlapping nodes (keep highest confidence)
        merged = self._merge_overlapping(quick_nodes)

        # Sort by message position
        merged.sort(key=lambda n: n.msg_start)

        # Convert to ReasoningNodes
        nodes = []
        for i, qn in enumerate(merged):
            nodes.append(ReasoningNode(
                node_id=f"warm-{i:03d}",
                node_type=qn.node_type,
                summary=qn.summary,
                evidence=qn.evidence,
                message_range=(qn.msg_start, qn.msg_end),
                confidence=qn.confidence,
            ))

        # Sequential edges
        edges = [
            ReasoningEdge(
                source_id=nodes[i].node_id,
                target_id=nodes[i + 1].node_id,
                relationship=self._infer_edge_type(nodes[i], nodes[i + 1]),
            )
            for i in range(len(nodes) - 1)
        ]

        pivots = [n.node_id for n in nodes if n.node_type in (NodeType.PIVOT, NodeType.DEAD_END)]

        original_count = len(session.messages)
        noise_ratio = 1.0 - (len(nodes) / original_count) if original_count else 0.0

        return ReasoningDAG(
            session_id=session.session_id,
            nodes=nodes,
            edges=edges,
            pivot_nodes=pivots,
            noise_ratio=noise_ratio,
        )

    # ── Strategy 1: Keyword classification ────────────────────────

    def _classify_messages(self, session: NormalizedSession) -> list[_QuickNode]:
        """Classify each assistant message by keyword patterns."""
        nodes = []
        for i, msg in enumerate(session.messages):
            if msg.role != MessageRole.ASSISTANT:
                continue
            if len(msg.content) < self.min_message_length:
                continue

            scores = {
                NodeType.HYPOTHESIS: _match_score(msg.content, _HYPOTHESIS_PATTERNS),
                NodeType.DISCOVERY: _match_score(msg.content, _DISCOVERY_PATTERNS),
                NodeType.PIVOT: _match_score(msg.content, _PIVOT_PATTERNS),
                NodeType.SOLUTION: _match_score(msg.content, _SOLUTION_PATTERNS),
                NodeType.DEAD_END: _match_score(msg.content, _DEAD_END_PATTERNS),
            }

            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]

            if best_score == 0:
                # No keywords matched — context load or investigation
                if msg.files_referenced or msg.files_modified:
                    ntype = NodeType.INVESTIGATION
                    conf = 0.35
                else:
                    ntype = NodeType.CONTEXT_LOAD
                    conf = 0.25
            else:
                ntype = best_type
                # Confidence scales with number of matching patterns
                conf = min(0.7, 0.35 + best_score * 0.1)

            summary = msg.content[:200].strip()
            if len(msg.content) > 200:
                summary = summary[:197] + "..."

            nodes.append(_QuickNode(
                node_type=ntype,
                summary=summary,
                evidence=f"keyword classification ({best_score} pattern matches)",
                msg_start=i,
                msg_end=i + 1,
                confidence=conf,
                source="keyword",
            ))

        return nodes

    # ── Strategy 2: Error-resolution pairs ────────────────────────

    def _find_error_resolutions(self, session: NormalizedSession) -> list[_QuickNode]:
        """Find error messages followed by successful resolutions."""
        nodes = []
        messages = session.messages

        for i, msg in enumerate(messages):
            # Look for error indicators in any message
            if not _match_score(msg.content, _ERROR_PATTERNS):
                continue

            # Look ahead for resolution
            window_end = min(i + self.error_resolution_window, len(messages))
            for j in range(i + 1, window_end):
                resolve_msg = messages[j]
                if resolve_msg.role != MessageRole.ASSISTANT:
                    continue
                if _match_score(resolve_msg.content, _SOLUTION_PATTERNS):
                    error_snippet = msg.content[:100].strip()
                    resolve_snippet = resolve_msg.content[:100].strip()
                    nodes.append(_QuickNode(
                        node_type=NodeType.SOLUTION,
                        summary=f"Resolved error: {resolve_snippet}",
                        evidence=f"Error at msg {i}: {error_snippet}",
                        msg_start=i,
                        msg_end=j + 1,
                        confidence=0.6,
                        source="error_resolution",
                    ))
                    break

        return nodes

    # ── Strategy 3: Explicit conclusions ──────────────────────────

    def _find_explicit_conclusions(self, session: NormalizedSession) -> list[_QuickNode]:
        """Find messages where the agent states explicit conclusions."""
        nodes = []
        for i, msg in enumerate(session.messages):
            if msg.role != MessageRole.ASSISTANT:
                continue

            score = _match_score(msg.content, _CONCLUSION_PATTERNS)
            if score == 0:
                continue

            # Extract the conclusion sentence
            text = msg.content
            for pattern in _CONCLUSION_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Get the sentence containing the match
                    start = max(0, text.rfind(".", 0, match.start()) + 1)
                    end = text.find(".", match.end())
                    if end == -1:
                        end = min(len(text), match.end() + 150)
                    conclusion = text[start:end].strip()
                    break
            else:
                conclusion = text[:200]

            nodes.append(_QuickNode(
                node_type=NodeType.DISCOVERY,
                summary=conclusion,
                evidence="explicit conclusion statement",
                msg_start=i,
                msg_end=i + 1,
                confidence=0.65,
                source="conclusion",
            ))

        return nodes

    # ── Merging and edge inference ────────────────────────────────

    def _merge_overlapping(self, nodes: list[_QuickNode]) -> list[_QuickNode]:
        """Merge nodes that overlap in message range, keeping highest confidence."""
        if not nodes:
            return []

        # Sort by start position
        nodes.sort(key=lambda n: (n.msg_start, -n.confidence))

        merged = [nodes[0]]
        for node in nodes[1:]:
            prev = merged[-1]
            # Check overlap
            if node.msg_start <= prev.msg_end and node.source != prev.source:
                # Different heuristics found the same region — keep higher confidence
                if node.confidence > prev.confidence:
                    merged[-1] = node
                # Boost confidence of the winner slightly for corroboration
                merged[-1].confidence = min(0.75, merged[-1].confidence + 0.05)
            elif node.msg_start == prev.msg_start and node.source == prev.source:
                # Same heuristic, same position — skip duplicate
                continue
            else:
                merged.append(node)

        return merged

    @staticmethod
    def _infer_edge_type(source: ReasoningNode, target: ReasoningNode) -> str:
        """Infer edge relationship from node type transitions."""
        if target.node_type == NodeType.PIVOT:
            return "triggered_pivot"
        if source.node_type == NodeType.DEAD_END:
            return "triggered_pivot" if target.node_type == NodeType.PIVOT else "led_to"
        if target.node_type == NodeType.DISCOVERY:
            return "revealed"
        if source.node_type == NodeType.HYPOTHESIS:
            return "informed"
        if target.node_type == NodeType.SOLUTION:
            return "enabled"
        return "led_to"
