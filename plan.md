# Implementation Plan: Persistent Cross-Platform Cognitive Memory for AI Coding Agents

## Guiding Principles

This plan follows three rules:

1. **Every phase produces something testable.** No phase exists purely as "infrastructure." Each one ends with a concrete demo you can run and show to someone.
2. **Start with one platform, one codebase, one memory type.** Resist the urge to build the universal parser first. Build the narrowest useful thing, prove it works, then generalize.
3. **The pipeline is the product.** The novel contribution is the extraction→compression→delivery pipeline. Every implementation decision should serve that pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                              │
│  Claude Code Transcripts → Parser → Normalized Session Format       │
│  (Later: Cursor, Copilot, Codex CLI, etc.)                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTRACTION ENGINE                                │
│  Normalized Session → LLM Analysis → Reasoning DAG                  │
│  Pivot Detection · Noise Filtering · Insight Identification         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  COMPRESSION & CONSOLIDATION                        │
│  Embedding · Clustering · LLM Merge Decisions → Cognitive Profiles  │
│  Dedup · Conflict Resolution · Profile Updates                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MEMORY STORE                                   │
│  Hot: In-session retrieval (vector DB)                              │
│  Warm: Between-session heuristics (cached profiles)                 │
│  Cold: Batch consolidation (full reprocessing)                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DELIVERY LAYER (MCP)                            │
│  MCP Server exposing: search_memory, get_profile, get_pitfalls     │
│  Any MCP-compatible agent can connect with zero modifications       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Foundations (Week 1)

**Goal:** Set up the project, collect raw material, define the data formats that everything else builds on.

### 0.1 Project Scaffolding

```
cognitive-memory/
├── README.md
├── plan.md                  # this file
├── pyproject.toml           # Python project (uv or poetry)
├── src/
│   ├── ingestion/           # session parsers
│   ├── extraction/          # DAG builder, pivot detection
│   ├── compression/         # embedding, clustering, merging
│   ├── store/               # memory storage layer
│   ├── delivery/            # MCP server
│   └── schemas/             # shared data models
├── tests/
├── fixtures/                # sample sessions for testing
│   ├── claude_code/
│   ├── cursor/
│   └── synthetic/
└── scripts/                 # utility scripts
```

**Tech stack decisions (start lean, expand later):**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12+ | Richest LLM ecosystem, fastest prototyping |
| LLM calls | Anthropic SDK (Claude Sonnet) | Best cost/quality for structured extraction |
| Embeddings | `sentence-transformers` (local) or OpenAI `text-embedding-3-small` | Start local, switch to API if quality demands |
| Vector DB | ChromaDB (local, file-based) | Zero infrastructure, good enough to Phase 3 |
| Data models | Pydantic v2 | Strict validation, JSON serialization, schema evolution |
| MCP server | `mcp` Python SDK | Official SDK, minimal boilerplate |
| Testing | pytest + inline fixtures | Fast iteration |

### 0.2 Collect Raw Session Data

This is the most important step in Phase 0. You need real sessions to build against.

**Action items:**

- [ ] Export 5–10 Claude Code session transcripts from your own coding work. These live in `~/.claude/projects/` as JSONL files. Copy them to `fixtures/claude_code/`.
- [ ] If you use Cursor, export 3–5 conversation logs. Cursor stores conversations in its SQLite database at `~/Library/Application Support/Cursor/User/workspaceStorage/`.
- [ ] Create 2–3 synthetic sessions by hand that represent known patterns: a debugging session with a clear pivot, a refactoring session with architectural discovery, a session that goes nowhere (negative example for noise filtering).
- [ ] Document the format of each source in `fixtures/FORMAT_NOTES.md`.

### 0.3 Define Core Schemas

These Pydantic models are the contract between every component. Get them right early.

```python
# src/schemas/session.py
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

class SessionMessage(BaseModel):
    """Single message in a normalized session."""
    role: MessageRole
    content: str
    timestamp: datetime | None = None
    tool_name: str | None = None        # for tool calls/results
    files_referenced: list[str] = []     # files mentioned or edited
    files_modified: list[str] = []       # files actually changed

class NormalizedSession(BaseModel):
    """Platform-agnostic session representation."""
    session_id: str
    source_platform: str                 # "claude_code", "cursor", etc.
    project_path: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    messages: list[SessionMessage]
    raw_metadata: dict = {}              # platform-specific extras
```

```python
# src/schemas/reasoning.py
from pydantic import BaseModel
from enum import Enum

class NodeType(str, Enum):
    HYPOTHESIS = "hypothesis"       # agent forms a theory
    INVESTIGATION = "investigation" # agent examines evidence
    DISCOVERY = "discovery"         # agent finds something unexpected
    PIVOT = "pivot"                 # agent changes direction
    SOLUTION = "solution"           # agent reaches a resolution
    DEAD_END = "dead_end"           # path that didn't work
    CONTEXT_LOAD = "context_load"   # agent reads/understands code

class ReasoningNode(BaseModel):
    """Single node in the reasoning DAG."""
    node_id: str
    node_type: NodeType
    summary: str                        # 1-2 sentence description
    evidence: str                       # what triggered this node
    message_range: tuple[int, int]      # indices into session messages
    confidence: float = 0.0             # how certain the extraction is

class ReasoningEdge(BaseModel):
    """Directed edge between reasoning nodes."""
    source_id: str
    target_id: str
    relationship: str                   # "led_to", "contradicted", "refined", etc.

class ReasoningDAG(BaseModel):
    """Directed acyclic graph of an agent's reasoning trajectory."""
    session_id: str
    nodes: list[ReasoningNode]
    edges: list[ReasoningEdge]
    pivot_nodes: list[str] = []         # node_ids where direction changed
    noise_ratio: float = 0.0           # fraction of session filtered out
```

```python
# src/schemas/memory.py
from pydantic import BaseModel
from datetime import datetime

class Pitfall(BaseModel):
    description: str
    frequency: int = 1                   # how many sessions encountered this
    severity: str = "medium"             # low, medium, high
    resolution_strategy: str | None = None

class DiagnosticStrategy(BaseModel):
    trigger: str                         # when to use this strategy
    steps: list[str]                     # ordered diagnostic steps
    success_rate: float = 0.0            # how often this works
    source_sessions: list[str] = []      # which sessions derived this

class ArchitecturalInsight(BaseModel):
    component: str                       # which part of the codebase
    insight: str                         # what the agent learned
    confidence: float = 0.0

class CognitiveProfile(BaseModel):
    """The consolidated output: what an agent should know about this codebase."""
    project_id: str
    last_updated: datetime
    architectural_insights: list[ArchitecturalInsight] = []
    pitfalls: list[Pitfall] = []
    diagnostic_strategies: list[DiagnosticStrategy] = []
    key_patterns: list[str] = []         # recurring patterns observed
    anti_patterns: list[str] = []        # things that consistently fail
    session_count: int = 0               # how many sessions contributed
```

**Phase 0 exit criteria:**

- [ ] Project structure exists with all directories
- [ ] Dependencies install cleanly
- [ ] At least 5 real session transcripts in `fixtures/`
- [ ] All Pydantic schemas validate with synthetic data
- [ ] `pytest` runs (even if only schema validation tests)

---

## Phase 1: Single-Platform Extraction (Weeks 2–3)

**Goal:** Ingest one Claude Code session, extract a reasoning DAG, and print it. This is the core novel contribution — get it working on the narrowest possible case first.

### 1.1 Claude Code Session Parser

Claude Code stores transcripts as JSONL files. Each line is a message event with role, content, tool invocations, and timestamps.

```python
# src/ingestion/claude_code.py

class ClaudeCodeParser:
    """Parse Claude Code JSONL transcripts into NormalizedSession."""
    
    def parse_file(self, path: Path) -> NormalizedSession:
        # Read JSONL, extract messages, normalize roles
        # Identify tool calls (Read, Write, Bash, etc.)
        # Extract file references from tool arguments
        # Return NormalizedSession
        ...
```

**Key parsing decisions:**

- Tool calls and their results should be separate `SessionMessage` entries with `role=TOOL_CALL` and `role=TOOL_RESULT`. This preserves the action-observation structure that the DAG builder needs.
- File paths mentioned in `Read`, `Write`, `Edit` tool calls populate `files_referenced` and `files_modified`.
- Auto-compaction boundaries (where Claude Code summarizes context) should be preserved in `raw_metadata` — they indicate natural session segments.

**Test:** Parse a real transcript, print the `NormalizedSession`, verify message count and file lists match what you see in the raw JSONL.

### 1.2 Reasoning DAG Extractor

This is the hardest and most novel component. Start with a simple two-pass approach:

**Pass 1: Chunking and classification.** Break the session into overlapping windows of 3–5 messages. For each window, use an LLM to classify the dominant reasoning activity (hypothesis, investigation, discovery, pivot, dead_end, solution, context_load).

**Pass 2: Graph construction.** Take the classified chunks, ask the LLM to identify edges (what led to what), and detect pivot nodes (where the agent's direction changed).

```python
# src/extraction/dag_builder.py

class DAGBuilder:
    """Extract reasoning DAG from a normalized session."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def build(self, session: NormalizedSession) -> ReasoningDAG:
        # Step 1: Pre-filter noise (tool results that are just file contents,
        #         routine acknowledgments, etc.)
        filtered = self._prefilter(session.messages)
        
        # Step 2: Chunk into reasoning windows
        windows = self._create_windows(filtered, window_size=5, overlap=2)
        
        # Step 3: Classify each window (batch LLM calls)
        nodes = await self._classify_windows(windows)
        
        # Step 4: Build edges between nodes
        edges = await self._identify_edges(nodes)
        
        # Step 5: Detect pivots
        pivots = self._detect_pivots(nodes, edges)
        
        return ReasoningDAG(
            session_id=session.session_id,
            nodes=nodes,
            edges=edges,
            pivot_nodes=pivots,
            noise_ratio=1 - len(filtered) / len(session.messages)
        )
```

**The extraction prompt is the most important artifact in this project.** Start with something like:

```
You are analyzing a segment of a coding agent's session. Your task is to identify 
the reasoning activity in this segment.

Classify as ONE of:
- HYPOTHESIS: The agent forms a theory about what might be wrong or how to proceed
- INVESTIGATION: The agent examines code, runs tests, or gathers evidence
- DISCOVERY: The agent finds something unexpected that changes understanding
- PIVOT: The agent explicitly changes approach based on new information
- SOLUTION: The agent reaches a working resolution
- DEAD_END: The agent's current approach fails and is abandoned
- CONTEXT_LOAD: The agent is reading/understanding code without active reasoning

Respond with JSON:
{
  "node_type": "...",
  "summary": "1-2 sentence description of what the agent is doing and why",
  "evidence": "the specific message or observation that characterizes this segment",
  "confidence": 0.0-1.0
}
```

**Test:** Run extraction on 3 real sessions. Manually inspect the resulting DAGs. Are the node types reasonable? Do the pivots correspond to real direction changes? Iterate on the prompt until the output matches your intuition for at least 2 out of 3 sessions.

### 1.3 DAG Visualization

Build a simple visualization so you can inspect results. This is a development tool, not a product feature, but it's essential for iterating on extraction quality.

```python
# scripts/visualize_dag.py
# Use graphviz or mermaid to render the DAG
# Color-code by node type
# Bold the pivot nodes
# Label edges with relationship type
```

**Phase 1 exit criteria:**

- [ ] Claude Code parser produces valid `NormalizedSession` from real transcripts
- [ ] DAG builder produces valid `ReasoningDAG` from at least 3 real sessions
- [ ] Visual inspection confirms: pivots are real, noise filtering removes obvious filler, node types are mostly correct
- [ ] Noise ratio is in the 70–95% range (most of a session IS noise — file contents, routine tool calls, etc.)
- [ ] End-to-end test: `session.jsonl → NormalizedSession → ReasoningDAG → visualization`

---

## Phase 2: Memory Storage & First Cognitive Profile (Weeks 4–5)

**Goal:** Take multiple DAGs from the same project, compress and consolidate them into a single cognitive profile, store it, and retrieve it.

### 2.1 Embedding & Vector Storage

Embed each `ReasoningNode` summary using a sentence transformer. Store in ChromaDB with metadata (node type, session ID, project path, timestamp).

```python
# src/store/vector_store.py

class MemoryStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="reasoning_nodes",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def store_dag(self, dag: ReasoningDAG, project_id: str):
        # Embed and store each node with metadata
        ...

    def search(self, query: str, project_id: str, top_k: int = 5):
        # Semantic search within a project's memories
        ...
```

**Test:** Store 3 DAGs from the same project. Search for "authentication bug" or "database connection issue." Verify that relevant nodes surface from across sessions.

### 2.2 Semantic Deduplication

When storing a new DAG, check for near-duplicate nodes from previous sessions. This prevents the memory from growing unboundedly as more sessions are ingested.

```python
# src/compression/dedup.py

class SemanticDeduplicator:
    def __init__(self, store: MemoryStore, similarity_threshold: float = 0.85):
        self.store = store
        self.threshold = similarity_threshold
    
    def deduplicate(self, new_nodes: list[ReasoningNode], project_id: str):
        # For each new node, check similarity against existing nodes
        # If similarity > threshold, merge (LLM decides which to keep/how to combine)
        # If below threshold, store as new
        # Return: (stored_nodes, merged_nodes, dropped_nodes)
        ...
```

**Test:** Ingest two sessions where the agent encounters the same bug. Verify that the deduplicator recognizes the overlap and merges rather than duplicating.

### 2.3 Cognitive Profile Builder

This is the consolidation step: take all the reasoning nodes and edges for a project and distill them into a `CognitiveProfile`.

```python
# src/compression/profile_builder.py

class ProfileBuilder:
    """Consolidate reasoning memories into a structured cognitive profile."""
    
    async def build_profile(self, project_id: str, store: MemoryStore) -> CognitiveProfile:
        # 1. Retrieve all nodes for this project
        all_nodes = store.get_all_nodes(project_id)
        
        # 2. Cluster by embedding similarity
        clusters = self._cluster_nodes(all_nodes)
        
        # 3. For each cluster, use LLM to extract:
        #    - Is this an architectural insight, a pitfall, or a diagnostic strategy?
        #    - Summarize the cluster into the appropriate schema
        
        # 4. Rank pitfalls by frequency and severity
        # 5. Rank diagnostic strategies by success rate
        
        return CognitiveProfile(...)
```

**The profile-building prompt should produce structured output:**

```
Given these related reasoning fragments from coding sessions on the same project,
determine what type of knowledge they represent and consolidate:

[cluster of 3-5 node summaries]

Classify as ONE of:
- ARCHITECTURAL_INSIGHT: Something structural about how this codebase works
- PITFALL: A recurring problem or trap that agents fall into  
- DIAGNOSTIC_STRATEGY: A debugging/investigation approach that proved effective

Then provide a consolidated summary following the appropriate schema.
```

**Test:** Build a profile from 5+ sessions on one of your real projects. Read the output. Does it capture things you know to be true about the codebase? Would it be useful context for a new agent session?

**Phase 2 exit criteria:**

- [ ] Vector store persists across restarts
- [ ] Deduplication measurably reduces redundancy (track node count before/after)
- [ ] At least one cognitive profile built from real session data
- [ ] The profile contains at least: 2 architectural insights, 2 pitfalls, 1 diagnostic strategy
- [ ] Manual review confirms the profile is accurate and useful

---

## Phase 3: MCP Delivery — The First Working Demo (Weeks 6–7)

**Goal:** Serve the cognitive profile via MCP so that any compatible coding agent can access it. This is the milestone where the system becomes usable in practice.

### 3.1 MCP Server

Build an MCP server that exposes the memory store and cognitive profiles as tools.

```python
# src/delivery/mcp_server.py
from mcp.server import Server
from mcp.types import Tool

app = Server("cognitive-memory")

@app.tool()
async def search_memory(query: str, project_id: str = None) -> str:
    """Search cognitive memory for relevant reasoning patterns.
    
    Returns reasoning fragments from past coding sessions that are 
    semantically relevant to the query. Use this when you encounter 
    a problem or need to understand how this codebase works.
    """
    results = store.search(query, project_id, top_k=5)
    return format_results(results)

@app.tool()
async def get_cognitive_profile(project_id: str) -> str:
    """Get the full cognitive profile for a project.
    
    Returns accumulated knowledge: architectural insights, known 
    pitfalls, and proven diagnostic strategies. Use this at the start 
    of a session to understand the codebase context.
    """
    profile = store.get_profile(project_id)
    return format_profile(profile)

@app.tool()
async def get_pitfalls(project_id: str) -> str:
    """Get known pitfalls for a project, ranked by severity.
    
    Returns a ranked list of things that have caused problems in 
    past sessions. Check this before making changes to warn yourself 
    about common traps.
    """
    profile = store.get_profile(project_id)
    return format_pitfalls(profile.pitfalls)

@app.tool()
async def get_diagnostic_strategy(problem_description: str, project_id: str) -> str:
    """Find a proven diagnostic strategy for a problem type.
    
    Given a problem description, returns diagnostic approaches that 
    have worked in past sessions on this codebase.
    """
    strategies = store.search_strategies(problem_description, project_id)
    return format_strategies(strategies)
```

### 3.2 MCP Configuration

Provide config snippets for connecting the server to popular agents:

```jsonc
// Claude Code: ~/.claude/mcp_servers.json
{
  "cognitive-memory": {
    "command": "python",
    "args": ["-m", "cognitive_memory.delivery.mcp_server"],
    "env": {
      "MEMORY_STORE_PATH": "/path/to/memory/store"
    }
  }
}

// Cursor: settings.json MCP section (similar format)
// Claude Desktop: claude_desktop_config.json
```

### 3.3 The Demo

This is the demo you can show to people:

1. **Before:** Start a Claude Code session on a project. Ask it to investigate a bug that a previous agent already solved. Watch it stumble through the same investigation.
2. **Ingest:** Run the pipeline on 5 past session transcripts from that project.
3. **After:** Start a new Claude Code session on the same project, now with the MCP server connected. The agent can call `get_cognitive_profile` at session start and `search_memory` when it hits a problem. Watch it skip the dead ends and go straight to the proven diagnostic strategy.

**Phase 3 exit criteria:**

- [ ] MCP server starts and is discoverable by Claude Code
- [ ] All 4 tools return meaningful results
- [ ] End-to-end demo: past session → ingestion → extraction → storage → MCP retrieval in new session
- [ ] At least one instance where the memory visibly helps an agent avoid a known pitfall

---

## Phase 4: Warm Tier & Background Processing (Weeks 8–9)

**Goal:** Add the between-session processing tier so that ingestion happens automatically, not manually.

### 4.1 Session Watcher

A lightweight daemon that watches for new completed sessions and triggers extraction.

```python
# src/ingestion/watcher.py

class SessionWatcher:
    """Watch for completed coding sessions and trigger ingestion."""
    
    def __init__(self, watch_paths: list[Path], pipeline: ExtractionPipeline):
        self.watch_paths = watch_paths  # e.g., ~/.claude/projects/
        self.pipeline = pipeline
        self.processed: set[str] = set()  # track what we've already ingested
    
    async def watch(self):
        # Poll for new/modified session files
        # When detected, run quick heuristic extraction (warm tier)
        # Queue full extraction for cold tier batch processing
        ...
```

### 4.2 Warm-Tier Heuristic Extraction

The warm tier needs to run in under a second. It can't afford full LLM-based DAG extraction. Instead, it uses fast heuristics:

```python
# src/extraction/warm_extractor.py

class WarmExtractor:
    """Fast heuristic extraction for between-session processing."""
    
    def extract_quick(self, session: NormalizedSession) -> list[QuickMemory]:
        memories = []
        
        # Heuristic 1: Error-resolution pairs
        # Find messages containing errors followed by successful resolutions
        memories.extend(self._find_error_resolutions(session))
        
        # Heuristic 2: File modification patterns  
        # Which files were modified together? (architectural coupling signal)
        memories.extend(self._find_modification_patterns(session))
        
        # Heuristic 3: Repeated attempts
        # Same command/action attempted multiple times = difficulty signal
        memories.extend(self._find_repeated_attempts(session))
        
        # Heuristic 4: Explicit agent statements
        # "I found that...", "The issue was...", "This approach works..."
        memories.extend(self._find_explicit_conclusions(session))
        
        return memories
```

**Test:** Time the warm extractor on 10 real sessions. Verify all complete in under 1 second. Compare the quick memories to the full DAG extraction — how much signal does the warm tier capture vs. the cold tier?

### 4.3 Cold-Tier Batch Processing

Runs periodically (e.g., nightly) or on-demand. Performs full DAG extraction on all sessions that only got warm-tier processing, then rebuilds cognitive profiles.

```python
# scripts/batch_consolidate.py

async def batch_consolidate(project_id: str):
    """Full cold-tier consolidation for a project."""
    # 1. Find all sessions with only warm-tier processing
    # 2. Run full DAG extraction on each
    # 3. Deduplicate against existing memory
    # 4. Rebuild cognitive profile
    # 5. Update the memory store
```

**Phase 4 exit criteria:**

- [ ] Session watcher detects new Claude Code sessions within 5 seconds of completion
- [ ] Warm extraction completes in < 1 second for sessions up to 200 messages
- [ ] Cold batch processing rebuilds profiles correctly
- [ ] The three tiers produce consistent results (warm is a subset of cold, hot serves both)

---

## Phase 5: Second Platform & Cross-Platform Validation (Weeks 10–12)

**Goal:** Add Cursor as a second ingestion source. Prove that memories from Claude Code sessions are useful to Cursor agents and vice versa.

### 5.1 Cursor Session Parser

```python
# src/ingestion/cursor.py

class CursorParser:
    """Parse Cursor conversation logs into NormalizedSession."""
    # Cursor stores conversations differently than Claude Code
    # May need to read from SQLite workspace storage
    # The key is producing the same NormalizedSession schema
    ...
```

### 5.2 Cross-Platform Validation

The critical test: does a cognitive profile built from Claude Code sessions actually help a Cursor agent (or vice versa)?

**Experiment design:**

1. Pick a project with 5+ Claude Code sessions.
2. Build a cognitive profile from those sessions alone.
3. Start a Cursor session on the same project with the MCP server connected.
4. Give Cursor a task that previous Claude Code sessions struggled with.
5. Measure: Does the agent find the solution faster? Does it avoid known pitfalls?

**Qualitative evaluation criteria:**

- Does the agent reference memory results in its reasoning?
- Does it avoid dead ends that previous agents explored?
- Are the architectural insights from Claude Code sessions applicable in Cursor's context?

### 5.3 Parser Abstraction

With two parsers working, refactor to a clean plugin architecture:

```python
# src/ingestion/registry.py

class ParserRegistry:
    _parsers: dict[str, type[SessionParser]] = {}
    
    @classmethod
    def register(cls, platform: str):
        def decorator(parser_class):
            cls._parsers[platform] = parser_class
            return parser_class
        return decorator
    
    @classmethod
    def parse(cls, path: Path) -> NormalizedSession:
        platform = cls._detect_platform(path)
        parser = cls._parsers[platform]()
        return parser.parse_file(path)
```

**Phase 5 exit criteria:**

- [ ] Cursor parser produces valid `NormalizedSession` objects
- [ ] Cognitive profiles contain memories from both platforms
- [ ] At least one qualitative demonstration of cross-platform memory transfer
- [ ] Parser registry cleanly supports adding new platforms

---

## Phase 6: Evaluation & Hardening (Weeks 13–15)

**Goal:** Quantify the system's value and address the weaknesses identified during development.

### 6.1 Extraction Quality Evaluation

Create a small hand-labeled evaluation set:

- [ ] Take 10 sessions, manually annotate pivots and key insights
- [ ] Run the extractor, compare against gold labels
- [ ] Measure: pivot detection precision/recall, noise filter accuracy, node type accuracy
- [ ] Iterate on prompts until extraction quality is acceptable

### 6.2 Noise Ratio Validation

The abstract claims ~90% noise filtering. Validate this:

- [ ] For 10 sessions, manually classify each message as "signal" or "noise"
- [ ] Compare against the extractor's filtering decisions
- [ ] Report the actual noise ratio with confidence intervals

### 6.3 Memory Quality Evaluation

Does the cognitive profile actually contain correct information?

- [ ] For 3 projects, review profiles against your own knowledge of the codebase
- [ ] Flag: correct insights, incorrect insights, missing insights
- [ ] Calculate precision (what fraction of stored memories are correct)
- [ ] Calculate coverage (what fraction of important patterns are captured)

### 6.4 Profile Staleness & Conflict Resolution

As more sessions are ingested, earlier insights may become outdated. Implement:

- [ ] Temporal decay: older memories have lower weight in profile construction
- [ ] Conflict detection: when a new session contradicts an existing insight
- [ ] Profile versioning: track how profiles change over time

**Phase 6 exit criteria:**

- [ ] Quantitative extraction quality metrics on 10+ sessions
- [ ] Validated noise ratio with evidence
- [ ] Cognitive profile accuracy assessment
- [ ] Staleness handling implemented and tested

---

## Phase 7: Talk Preparation & Polish (Weeks 16+)

**Goal:** Package everything for the presentation described in the abstract.

### 7.1 Demo Materials

- [ ] Live demo: ingest 3 sessions → build profile → show in MCP → use in new session
- [ ] Before/after comparison video or screenshots
- [ ] DAG visualization of a real debugging session showing the pivot point

### 7.2 Quantitative Results for Slides

- [ ] Noise filtering ratio across N sessions
- [ ] Deduplication compression ratio (memories stored vs. raw session volume)
- [ ] Warm-tier latency measurements
- [ ] Cross-platform transfer example

### 7.3 Open Source Packaging

- [ ] Clean up codebase for public release
- [ ] Write installation and configuration docs
- [ ] Publish MCP server to a package registry
- [ ] Create a "quickstart" that gets someone from zero to working in 15 minutes

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM extraction quality is poor | High — the whole system depends on it | Invest heavily in prompt engineering in Phase 1. Build the eval set early. Consider using Claude Opus for extraction if Sonnet isn't sufficient. |
| Cognitive profiles are too generic | Medium — reduces practical value | Include concrete code references (file paths, function names) in profiles, not just abstract descriptions. |
| Warm tier can't meet sub-second latency | Medium — degrades developer experience | Pre-compute embeddings for common patterns. Use pure heuristics (regex, keyword matching) instead of LLM calls for the warm tier. |
| Session formats change across tool versions | Medium — breaks parsers | Pin to specific format versions. Use fuzzy parsing that tolerates missing fields. Test parsers on each tool update. |
| Cross-platform memories aren't useful | High — undermines core thesis | Validate early in Phase 5. If Claude Code memories don't help Cursor, the framing needs to shift to within-platform persistence. |
| Memory grows unboundedly | Low — solvable | Implement hard limits on profile size. Aggressive dedup. Periodic garbage collection of low-confidence memories. |

---

## Decision Log

Record key decisions here as you go. Format: `[date] Decision: ... Rationale: ...`

```
[YYYY-MM-DD] Decision: Start with Claude Code as first platform.
Rationale: Richest transcript format (includes tool calls, timestamps, 
file operations). Most accessible session data. Personal daily driver.

[YYYY-MM-DD] Decision: Use ChromaDB instead of Postgres/pgvector.
Rationale: Zero infrastructure overhead. File-based persistence. 
Can migrate to pgvector later if scale demands it.

[YYYY-MM-DD] Decision: Pydantic schemas defined before any implementation.
Rationale: Schemas are the contract between all components. Getting them 
right early prevents cascading refactors.
```
