"""Reasoning DAG extractor — two-pass LLM-based extraction."""
import asyncio
import json
import uuid
from typing import Any

import anthropic

from ..schemas.session import MessageRole, NormalizedSession, SessionMessage
from ..schemas.reasoning import NodeType, ReasoningDAG, ReasoningEdge, ReasoningNode

# ── Constants ────────────────────────────────────────────────────────────────

_WINDOW_SIZE = 5
_OVERLAP = 2
_MODEL = "claude-sonnet-4-6"

# Roles that are primarily noise (raw tool output)
_NOISE_ROLES = {MessageRole.TOOL_RESULT}

# Long tool result content is noise — cap at this many chars for context
_MAX_TOOL_RESULT_CHARS = 500

_CLASSIFICATION_SYSTEM = """\
You are an expert analyst of AI coding agent sessions. You identify the reasoning \
activity happening in a segment of an agent conversation.
"""

_CLASSIFICATION_PROMPT = """\
Analyze this segment of a coding agent's session and classify the dominant \
reasoning activity.

Classify as ONE of:
- HYPOTHESIS: The agent forms a theory about what might be wrong or how to proceed
- INVESTIGATION: The agent examines code, runs tests, or gathers evidence
- DISCOVERY: The agent finds something unexpected that changes understanding
- PIVOT: The agent explicitly changes approach based on new information
- SOLUTION: The agent reaches a working resolution
- DEAD_END: The agent's current approach fails and is abandoned
- CONTEXT_LOAD: The agent is reading/understanding code without active reasoning

Session segment:
{segment}

Respond with JSON only (no markdown, no explanation):
{{
  "node_type": "...",
  "summary": "1-2 sentence description of what the agent is doing and why",
  "evidence": "the specific message or observation that characterizes this segment",
  "confidence": 0.0-1.0
}}"""

_EDGE_SYSTEM = """\
You are an expert at understanding how reasoning steps connect in AI agent sessions.
"""

_EDGE_PROMPT = """\
Given these reasoning nodes extracted from a coding agent session, identify which \
nodes led to which. Return ONLY the edges that are clearly supported by the node summaries.

Nodes:
{nodes_json}

For each edge, the relationship should be one of: \
"led_to", "contradicted", "refined", "resolved", "discovered_from", "caused_pivot_to"

Respond with JSON only:
{{
  "edges": [
    {{"source_id": "...", "target_id": "...", "relationship": "..."}}
  ]
}}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_message(msg: SessionMessage, index: int) -> str:
    """Format a single message for inclusion in a prompt."""
    role_label = msg.role.value.upper()
    content = msg.content

    if msg.role == MessageRole.TOOL_CALL:
        tool = msg.tool_name or "unknown"
        try:
            inp = json.loads(content)
            # Summarize large inputs
            for k in ("content", "command"):
                if k in inp and isinstance(inp[k], str) and len(inp[k]) > 200:
                    inp[k] = inp[k][:200] + "..."
            content = f"[{tool}] {json.dumps(inp)}"
        except Exception:
            content = f"[{tool}] {content[:200]}"

    elif msg.role == MessageRole.TOOL_RESULT:
        content = content[:_MAX_TOOL_RESULT_CHARS]
        if len(msg.content) > _MAX_TOOL_RESULT_CHARS:
            content += "... [truncated]"

    return f"[{index}] {role_label}: {content}"


def _prefilter(messages: list[SessionMessage]) -> list[SessionMessage]:
    """Remove obvious noise: empty messages, pure file-dump tool results, etc."""
    filtered = []
    for msg in messages:
        if not msg.content.strip():
            continue
        # Very long tool results with no surrounding context = raw file dump
        if msg.role == MessageRole.TOOL_RESULT and len(msg.content) > 5000:
            # Keep a truncated version so the window has context
            short = SessionMessage(
                role=msg.role,
                content=msg.content[:_MAX_TOOL_RESULT_CHARS] + "... [truncated]",
                timestamp=msg.timestamp,
                tool_name=msg.tool_name,
            )
            filtered.append(short)
        else:
            filtered.append(msg)
    return filtered


def _create_windows(
    messages: list[SessionMessage], window_size: int = _WINDOW_SIZE, overlap: int = _OVERLAP
) -> list[tuple[int, int, list[SessionMessage]]]:
    """Slice messages into overlapping windows. Returns (start_idx, end_idx, messages)."""
    windows = []
    step = window_size - overlap
    i = 0
    while i < len(messages):
        end = min(i + window_size, len(messages))
        windows.append((i, end, messages[i:end]))
        if end == len(messages):
            break
        i += step
    return windows


async def _classify_window(
    client: anthropic.AsyncAnthropic,
    window_idx: int,
    start: int,
    end: int,
    msgs: list[SessionMessage],
) -> ReasoningNode | None:
    """Classify a single window via LLM. Returns None if extraction fails."""
    segment = "\n".join(_format_message(m, start + i) for i, m in enumerate(msgs))
    prompt = _CLASSIFICATION_PROMPT.format(segment=segment)

    try:
        response = await client.messages.create(
            model=_MODEL,
            max_tokens=256,
            system=_CLASSIFICATION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip possible markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data: dict[str, Any] = json.loads(raw)

        node_type_str = data.get("node_type", "CONTEXT_LOAD").upper()
        try:
            node_type = NodeType(node_type_str.lower())
        except ValueError:
            node_type = NodeType.CONTEXT_LOAD

        return ReasoningNode(
            node_id=f"node-{window_idx:03d}",
            node_type=node_type,
            summary=data.get("summary", ""),
            evidence=data.get("evidence", ""),
            message_range=(start, end),
            confidence=float(data.get("confidence", 0.5)),
        )
    except Exception as e:
        # Extraction failed for this window — return a low-confidence context_load
        return ReasoningNode(
            node_id=f"node-{window_idx:03d}",
            node_type=NodeType.CONTEXT_LOAD,
            summary=f"Window {window_idx}: extraction failed ({type(e).__name__})",
            evidence="",
            message_range=(start, end),
            confidence=0.0,
        )


async def _build_edges(
    client: anthropic.AsyncAnthropic, nodes: list[ReasoningNode]
) -> list[ReasoningEdge]:
    """Ask the LLM to identify edges between classified nodes."""
    if len(nodes) < 2:
        return []

    nodes_json = json.dumps(
        [
            {
                "node_id": n.node_id,
                "node_type": n.node_type.value,
                "summary": n.summary,
            }
            for n in nodes
        ],
        indent=2,
    )
    prompt = _EDGE_PROMPT.format(nodes_json=nodes_json)

    try:
        response = await client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_EDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)

        edges = []
        node_ids = {n.node_id for n in nodes}
        for e in data.get("edges", []):
            src = e.get("source_id", "")
            tgt = e.get("target_id", "")
            rel = e.get("relationship", "led_to")
            if src in node_ids and tgt in node_ids and src != tgt:
                edges.append(ReasoningEdge(source_id=src, target_id=tgt, relationship=rel))
        return edges

    except Exception:
        # Fallback: sequential edges
        return [
            ReasoningEdge(source_id=nodes[i].node_id, target_id=nodes[i + 1].node_id, relationship="led_to")
            for i in range(len(nodes) - 1)
        ]


def _detect_pivots(nodes: list[ReasoningNode], edges: list[ReasoningEdge]) -> list[str]:
    """Return node_ids of pivot or dead_end nodes."""
    pivot_types = {NodeType.PIVOT, NodeType.DEAD_END}
    return [n.node_id for n in nodes if n.node_type in pivot_types]


# ── Main class ───────────────────────────────────────────────────────────────

class DAGBuilder:
    """Extract a reasoning DAG from a normalized session using LLM analysis."""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def build(self, session: NormalizedSession) -> ReasoningDAG:
        original_count = len(session.messages)

        # Step 1: Filter noise
        filtered = _prefilter(session.messages)

        # Step 2: Create overlapping windows
        windows = _create_windows(filtered)

        # Step 3: Classify each window concurrently (batch LLM calls)
        tasks = [
            _classify_window(self.client, idx, start, end, msgs)
            for idx, (start, end, msgs) in enumerate(windows)
        ]
        nodes_raw = await asyncio.gather(*tasks)
        nodes = [n for n in nodes_raw if n is not None]

        # Step 4: Build edges
        edges = await _build_edges(self.client, nodes)

        # Step 5: Detect pivots
        pivots = _detect_pivots(nodes, edges)

        noise_ratio = 1.0 - (len(filtered) / original_count) if original_count > 0 else 0.0

        return ReasoningDAG(
            session_id=session.session_id,
            nodes=nodes,
            edges=edges,
            pivot_nodes=pivots,
            noise_ratio=noise_ratio,
        )
