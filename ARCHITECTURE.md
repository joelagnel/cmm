# Cognitive Memory Manager — Architecture

> Persistent reasoning memory for AI coding agents.
> Capture session intelligence, consolidate it into durable knowledge,
> and share it across developers through a human-gated distributed store.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow](#data-flow)
3. [Core Schemas](#core-schemas)
4. [Ingestion Pipeline](#ingestion-pipeline)
5. [Extraction Pipeline](#extraction-pipeline)
6. [Compression & Deduplication](#compression--deduplication)
7. [Storage Layer](#storage-layer)
8. [Delivery Layer](#delivery-layer)
9. [Project Discovery](#project-discovery)
10. [Evaluation System](#evaluation-system)
11. [Distributed Sync](#distributed-sync)
12. [CLI Reference](#cli-reference)
13. [Configuration](#configuration)
14. [Module Dependency Graph](#module-dependency-graph)
15. [Test Coverage](#test-coverage)

---

## System Overview

CMM processes Claude Code session transcripts (`.jsonl` files) through a
multi-stage pipeline that extracts reasoning patterns, deduplicates them
against stored knowledge, and consolidates them into a `CognitiveProfile`
— a structured representation of architectural insights, known pitfalls,
and proven diagnostic strategies for a project.

The system supports two operating modes:

- **LOCAL mode** (default): single developer, single ChromaDB instance.
  All reads and writes are local. No configuration beyond `cmm init`.

- **SHARED mode**: multiple developers sharing a second ChromaDB instance
  on a shared filesystem. New memories are staged, human-reviewed, then
  promoted to the shared main collection. Developers pull approved
  memories into their local cache.

### Tech Stack

| Component            | Technology                           |
| -------------------- | ------------------------------------ |
| Language             | Python 3.12+                         |
| Data validation      | Pydantic v2                          |
| Vector storage       | ChromaDB (local persistent)          |
| Embeddings           | OpenAI `text-embedding-3-small`      |
| LLM extraction       | Anthropic Claude Sonnet 4.5          |
| MCP delivery         | FastMCP (`mcp` v1.26.0)             |
| CLI framework        | Click                                |
| Terminal UI          | Rich                                 |
| Clustering           | scikit-learn (AgglomerativeClustering)|
| Session watching     | watchdog                             |
| Testing              | pytest + pytest-asyncio              |

---

## Data Flow

### Real-Time Pipeline (session end, < 5 seconds)

```
Claude Code session ends
  → ClaudeCodeParser.parse_file()          # src/ingestion/claude_code.py
    → NormalizedSession
  → WarmExtractor.extract()                # src/extraction/warm_extractor.py
    → ReasoningDAG (confidence 0.25–0.75)
  → SemanticDeduplicator.deduplicate()     # src/compression/dedup.py
    → DeduplicationResult
  → MemoryStore.store_dag()                # src/store/vector_store.py
    → nodes persisted in ChromaDB
  → SessionAnalyzer.analyze()              # src/evaluation/analyzer.py
    → evaluation metrics to SQLite
```

### Batch Pipeline (on demand or nightly)

```
cmm consolidate --project <id>
  → [optional] DAGBuilder re-extracts warm nodes with LLM
    → ReasoningDAG (confidence 0.5–0.95)
  → ProfileBuilder.build_profile()         # src/compression/profile_builder.py
    → cluster nodes → classify clusters → extract patterns
    → CognitiveProfile
  → MemoryStore.save_profile()
  → cached_profile.md updated
```

### Distributed Pipeline (shared mode)

```
Developer A: cmm push
  → local unpushed nodes → shared staging collection

Reviewer: cmm review
  → staging nodes → [approve | reject | reclassify | edit]
  → approved nodes → shared main collection

Developer B: cmm pull
  → shared main (approved project + team scope) → local cache
```

### Retrieval Pipeline (session start, < 500ms)

```
Claude Code session starts
  → session_start_hook reads .cmm/cached_profile.md
  → MemoryStore.search() with project llms.txt as query
  → inject profile + relevant memories into agent context
```

---

## Core Schemas

### `src/schemas/session.py`

Platform-agnostic session representation.

```python
class MessageRole(str, Enum):
    USER | ASSISTANT | SYSTEM | TOOL_CALL | TOOL_RESULT

class SessionMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime | None
    tool_name: str | None
    files_referenced: list[str]
    files_modified: list[str]

class NormalizedSession(BaseModel):
    session_id: str
    source_platform: str          # "claude_code", "cursor", etc.
    project_path: str | None
    started_at: datetime
    ended_at: datetime | None
    messages: list[SessionMessage]
    raw_metadata: dict
```

### `src/schemas/reasoning.py`

Reasoning graph structure.

```python
class NodeType(str, Enum):
    HYPOTHESIS | INVESTIGATION | DISCOVERY | PIVOT
    SOLUTION | DEAD_END | CONTEXT_LOAD

class MemoryScope(str, Enum):
    PROJECT = "project"           # specific to one repo
    TEAM = "team"                 # applicable across repos

class ReasoningNode(BaseModel):
    node_id: str
    node_type: NodeType
    summary: str                  # 1-2 sentences
    evidence: str
    message_range: tuple[int, int]
    confidence: float             # 0.0–1.0
    scope: MemoryScope            # default PROJECT

class ReasoningEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str             # "led_to", "refined", "caused_pivot_to", etc.

class ReasoningDAG(BaseModel):
    session_id: str
    nodes: list[ReasoningNode]
    edges: list[ReasoningEdge]
    pivot_nodes: list[str]
    noise_ratio: float
```

### `src/schemas/memory.py`

Consolidated knowledge.

```python
class Pitfall(BaseModel):
    description: str
    frequency: int                # sessions encountered
    severity: str                 # "low" | "medium" | "high"
    resolution_strategy: str | None
    scope: MemoryScope

class DiagnosticStrategy(BaseModel):
    trigger: str
    steps: list[str]
    success_rate: float
    source_sessions: list[str]
    scope: MemoryScope

class ArchitecturalInsight(BaseModel):
    component: str
    insight: str
    confidence: float
    scope: MemoryScope

class CognitiveProfile(BaseModel):
    project_id: str
    last_updated: datetime
    architectural_insights: list[ArchitecturalInsight]
    pitfalls: list[Pitfall]
    diagnostic_strategies: list[DiagnosticStrategy]
    key_patterns: list[str]
    anti_patterns: list[str]
    session_count: int
```

---

## Ingestion Pipeline

### `src/ingestion/claude_code.py` — ClaudeCodeParser

Converts Claude Code `.jsonl` transcripts into `NormalizedSession`.

- Parses line-by-line JSON events: `file-history-snapshot`, `user`, `assistant`
- User events: content can be string or list (text + tool_result blocks)
- Assistant events: content array of text + tool_use blocks
- Extracts file references and modifications from tool arguments
- Detects `isMeta` (summaries) and `isSidechain` (alternate branches)
- Preserves absolute message indices for downstream LLM reference

### `src/ingestion/watcher.py` — SessionWatcher

Polling daemon that detects new or modified `.jsonl` files under
`~/.claude/projects/` and auto-triggers warm-tier ingestion.

- Maintains state in `.cmm_watcher_state.json`
- Configurable: `poll_interval` (default 10s), `min_file_age` (default 30s)
- Derives project ID from the Claude Code encoded directory path
- Auto-ingest workflow: parse → warm extract → deduplicate → store

---

## Extraction Pipeline

### `src/extraction/warm_extractor.py` — WarmExtractor

Fast heuristic extraction (< 1 second, zero API calls). Runs at every
session end via `cmm hook stop`.

**Three strategies, merged with corroboration boost:**

| Strategy                | What it detects                   | Confidence |
| ----------------------- | --------------------------------- | ---------- |
| Keyword classification  | Regex patterns per node type      | 0.35 + matches × 0.1 (max 0.70) |
| Error-resolution pairs  | Error → solution within 8 msgs    | 0.60 |
| Explicit conclusions    | "I found that...", "The issue was" | 0.65 |

Overlap merging: when two strategies flag the same message range, the
higher-confidence node wins and gets a +0.05 corroboration boost (max 0.75).

Edge inference is purely structural (type transitions): `DEAD_END → PIVOT`
becomes `"triggered_pivot"`, `HYPOTHESIS → *` becomes `"informed"`, etc.

Node IDs: `warm-NNN`.

### `src/extraction/dag_builder.py` — DAGBuilder (cold tier)

Token-budget LLM extraction. Uses Anthropic Claude Sonnet 4.5.

**Why token-budget windows instead of fixed message windows:**

The old approach (fixed 5-message windows, 2-message overlap) was:
- Inefficient: a 200-message session produced ~60 LLM calls
- Low quality: reasoning sequences were arbitrarily split mid-thought
- Wasteful: each call used only a fraction of the model's context capacity

The new approach:
1. **Pre-count tokens** for every message via `client.messages.count_tokens()`
   (concurrent — all counted in parallel)
2. **Compute budget**: `200k × 0.45 − output_reserve − prompt_overhead ≈ 85k tokens`
3. **Greedy packing**: fill each window to the budget, then overlap the last
   ~750 tokens into the next window
4. **Single-window fast path**: if the entire session fits in one window,
   a single LLM call sees the complete reasoning arc
5. **Multi-node extraction**: each LLM call returns a **list** of 1–12 nodes
   (not just one), so node density is preserved even with fewer windows

**The 45% fill ratio** is configurable via `CMM_CONTEXT_FILL_RATIO` and
backed by research:

| Paper | Finding |
| ----- | ------- |
| Chroma 2025 | All 18 frontier models degrade with increasing input length |
| Liu et al. 2024 ("Lost in the Middle") | 30%+ accuracy drops for mid-context info |
| Paulsen 2025 (MECW) | Effective context is drastically below advertised maximums |

**Key constants:**

```python
MODEL_CONTEXT_WINDOW = 200_000
DEFAULT_FILL_RATIO = 0.45
DEFAULT_OUTPUT_RESERVE = 4_000
DEFAULT_OVERLAP_TOKENS = 750
_MAX_NODES_PER_WINDOW = 12
```

**Overlap-aware deduplication:** since adjacent windows share trailing
content, the same reasoning step may be extracted by both. The
`_dedupe_overlapping_nodes()` method merges same-type nodes with
overlapping message ranges, keeping the higher-confidence version.

Node IDs: `node-WWW-NN` (window index, node index within window).

---

## Compression & Deduplication

### `src/compression/dedup.py` — SemanticDeduplicator

Prevents memory pollution by comparing new nodes against the existing
store before writing.

1. Embed new nodes: `"{node_type}: {summary}"`
2. Fetch all existing embeddings for the project
3. For each new node, find the most similar existing node (cosine)
4. If similarity ≥ 0.85 (configurable):
   - New confidence > existing → **merge** (upsert replaces existing)
   - Otherwise → **drop** (existing is better)
5. If similarity < 0.85 → **store** as new

Returns: `DeduplicationResult(stored=[], merged=[], dropped=[])`

### `src/compression/profile_builder.py` — ProfileBuilder

Consolidates all stored nodes for a project into a `CognitiveProfile`.

1. Fetch all nodes from ChromaDB with their embeddings
2. Normalize embeddings and cluster via **Agglomerative Clustering**
   (cosine distance, threshold 0.4, average linkage)
3. For each cluster ≥ 2 nodes, call Claude to classify:
   - Type: `ARCHITECTURAL_INSIGHT | PITFALL | DIAGNOSTIC_STRATEGY`
   - Scope: `PROJECT | TEAM`
   - Type-specific fields (component, severity, trigger, steps, etc.)
4. Separate pass: extract `key_patterns` and `anti_patterns` from all
   node summaries (up to 60)
5. Assemble and persist the `CognitiveProfile`

---

## Storage Layer

### `src/store/vector_store.py` — MemoryStore

Dual-mode ChromaDB wrapper. All embedding via OpenAI `text-embedding-3-small`
(1536 dimensions, batches ≤ 100).

**Collections:**

| Collection (local)      | Purpose                             |
| ----------------------- | ----------------------------------- |
| `reasoning_nodes`       | All extracted reasoning nodes        |
| `cognitive_profiles`    | Serialized CognitiveProfile per project |

| Collection (shared)     | Purpose                             |
| ----------------------- | ----------------------------------- |
| `reasoning_nodes`       | Approved nodes (served to queries)   |
| `reasoning_nodes_staging` | Pending review (human gate)        |
| `cognitive_profiles`    | Shared team profiles                 |
| `cmm_meta`              | Embedding model compatibility check  |

**Node metadata written by `store_dag()`:**

```python
{
    "project_id": str,
    "session_id": str,
    "node_id": str,
    "node_type": str,
    "confidence": float,
    "msg_start": int,
    "msg_end": int,
    "is_pivot": bool,
    "scope": "project" | "team",
    "pushed_at": str,          # ISO timestamp or "" if unpushed
    "approved": bool,
    "approved_by": str,
    "source_developer": str,
}
```

**Key methods:**

| Method | Purpose |
| ------ | ------- |
| `store_dag(dag, project_id)` | Embed + upsert nodes to local |
| `search(query, project_id, top_k, scope)` | Cosine search over nodes |
| `get_profile(project_id)` | Load CognitiveProfile from JSON |
| `save_profile(profile)` | Persist CognitiveProfile |
| `get_unpushed_nodes(project_id)` | Nodes with empty `pushed_at` |
| `mark_pushed(ids, when)` | Stamp `pushed_at` (idempotent) |
| `stage_to_shared(nodes, developer)` | Copy to shared staging |
| `list_pending_in_staging(project_id)` | Unapproved staged nodes |
| `promote_from_staging(ids, approver, ...)` | Move staging → main |
| `reject_in_staging(ids, reviewer, reason)` | Soft reject (kept for audit) |
| `list_approved_shared(project_id, include_team)` | Pull source |
| `upsert_pulled_nodes(nodes, project_id)` | Insert pulled nodes locally |

**Backwards compatibility:** the legacy `persist_dir=` kwarg still works
and maps to `local_path`. All existing aliases (`self.client`,
`self.nodes_col`, `self.profiles_col`) point to the local store.

**Embedding model safety:** on shared-mode init, the shared store records
which embedding model was used. Subsequent connections fail if the model
doesn't match — this prevents cross-machine vector incompatibility.

---

## Delivery Layer

### `src/delivery/mcp_server.py` — MCP Server

FastMCP stdio server exposing four tools:

| Tool | Purpose |
| ---- | ------- |
| `search_memory(query, project_id, top_k)` | Semantic search over nodes |
| `get_cognitive_profile(project_id)` | Full profile as markdown |
| `get_pitfalls(project_id)` | Severity-ranked pitfall list |
| `get_diagnostic_strategy(problem, project_id)` | Proven debugging steps |

Start: `cmm serve` or `python -m src.delivery`

Env vars: `CMM_STORE_PATH`, `CMM_PROJECT_ID`

### `src/delivery/cli_query.py` — CLI Query Interface

Same four tools as direct CLI commands, used by the `.claude/commands/`
skill files. Every invocation is timed and logged to
`data/eval/interactions.db` for evaluation metrics.

Commands: `profile`, `pitfalls`, `search <query>`, `diagnose <problem>`

### Seven Claude Code Skills (`.claude/commands/`)

| Skill | File | What it runs |
| ----- | ---- | ------------ |
| `/cognitive-profile` | `cognitive-profile.md` | `cli_query profile` |
| `/search-memory` | `search-memory.md` | `cli_query search "$ARGUMENTS"` |
| `/pitfalls` | `pitfalls.md` | `cli_query pitfalls` |
| `/diagnose` | `diagnose.md` | `cli_query diagnose "$ARGUMENTS"` |
| `/consolidate` | `consolidate.md` | `batch_consolidate.py --profiles-only` |
| `/visualize-dag` | `visualize-dag.md` | `visualize_dag.py` |
| `/watch-sessions` | `watch-sessions.md` | `cmm watch` |

---

## Project Discovery

### `src/discovery/project.py` — CognitiveProject

A `.cmm/` folder in the repo root makes a project self-describing.

```
.cmm/
├── manifest.json       # project_id, name, description, session_count
├── config.json         # auto_ingest, mode, shared_store_path, developer_name, etc.
├── cached_profile.md   # fast-load markdown snapshot of latest profile
└── llms.txt            # machine-readable metadata for agent consumption
```

**Project ID generation:** `{repo-name}-{sha256(README)[:6]}` — stable
across machines without hardcoding. Uses `git remote get-url origin` for
the repo name, falls back to the directory name.

**`discover_project(start_dir)`** walks up from `start_dir` looking for
`.cmm/manifest.json`, enabling automatic discovery from any
subdirectory.

### `src/discovery/hooks.py` — Session Hooks

| Hook | Trigger | What it does |
| ---- | ------- | ------------ |
| `session_start_hook` | Claude Code starts | Loads `cached_profile.md` + runs semantic search with project description; injects results into agent context |
| `session_stop_hook` | Claude Code ends | Finds latest JSONL, runs warm extraction, dedup, store, evaluation |

Designed to be wired into Claude Code's hook system:
```json
{"hooks": {"Stop": [{"command": "cmm hook stop"}]}}
```

### `src/discovery/llms_txt.py` — llms.txt Generation

Generates machine-readable project metadata including:
- Stack detection (scans for `pyproject.toml`, `package.json`, etc.)
- Known pitfalls with resolutions
- Diagnostic strategy triggers
- Retrieval hints for the agent

---

## Evaluation System

### `src/evaluation/logger.py` — InteractionLogger

SQLite-backed logging at `data/eval/interactions.db`. Two tables:

**`invocations`** — one row per skill call:
```
invocation_id, session_id, project_id, timestamp, skill,
query_text, result_count, node_ids[], similarity_scores[],
response_time_ms, raw_output_len
```

**`session_evaluations`** — one row per analyzed session:
```
session_id, project_id, evaluated_at, total_invocations,
memory_used_at_start, errors_encountered, errors_resolved_with_memory,
pitfalls_surfaced, pitfalls_avoided, pivots_after_retrieval,
total_dead_ends, total_pivots, messages_to_first_solution,
total_messages, total_nodes, duration_seconds
```

Fail-safe: every operation catches `Exception` and passes silently.
Logging never blocks retrieval (< 5ms overhead per invocation).

### `src/evaluation/analyzer.py` — SessionAnalyzer

Correlates session transcripts with interaction logs to measure
memory helpfulness via three signals:

**Signal A — Errors Resolved With Memory:**
`DEAD_END` at message X → retrieval invocation near X (±8 messages)
→ `SOLUTION` within X+8. The memory helped fix the error.

**Signal B — Pitfall Avoidance:**
`/pitfalls` or `/cognitive-profile` was called. For each stored pitfall,
check if a `DEAD_END` in the session matches it (≥ 3 word overlap). If no
match → the pitfall was **avoided**.

**Signal C — Pivots After Retrieval:**
A `PIVOT` node within 5 messages of a `/search-memory` or `/diagnose`
invocation. Memory influenced the direction change.

**`compare_sessions(baseline, assisted)`** computes percentage reductions
in dead ends, pivots, and time-to-solution between sessions with and
without memory.

### `scripts/eval_report.py` — Dashboard

Aggregates evaluation data into:
1. Retrieval metrics (avg similarity, avg response time)
2. Skill usage breakdown
3. Helpfulness signals (errors resolved, pitfalls avoided)
4. Memory vs no-memory session comparison

---

## Distributed Sync

### `src/sync/sync.py` — Syncer

Coordinates push/pull between local and shared ChromaDB stores.

**`push(project_id, dry_run=False) → PushResult`**

1. Query local nodes where `pushed_at == ""` (unpushed)
2. Copy their existing embeddings (no re-embedding) into shared staging
   with `approved=False` and `source_developer` attribution
3. Stamp `pushed_at` on local copies so they don't push again
4. Record event in `SyncLog`

**`pull(project_id, include_team=True) → PullResult`**

1. Fetch approved nodes from shared main for `project_id`
2. If `include_team`, also fetch all approved `scope="team"` nodes
   regardless of project
3. Filter out anything already present locally (by exact ID)
4. Upsert new nodes locally with `pushed_at` set (prevents bounce-back)
5. Record event in `SyncLog`

**`status(project_id) → dict`**

Returns: local node count, unpushed count, shared approved count,
pending review count, last push/pull timestamps, developer name.

### `SyncLog` — Audit Trail

SQLite database at `data/sync/sync.db`.

```sql
CREATE TABLE sync_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    action      TEXT NOT NULL,    -- 'push', 'pull', 'approve', 'reject'
    timestamp   TEXT NOT NULL,
    actor       TEXT,
    count       INTEGER,
    detail      TEXT
);
```

### `src/sync/review.py` — Reviewer

Headless review engine with a callback-based design that separates
business logic from UI.

**`review(project_id, decide_callback) → ReviewSummary`**

The `decide_callback` receives `(node_dict, current_index, total)` and
returns a `ReviewDecision`:

| Action | Effect |
| ------ | ------ |
| `APPROVE` | Promotes node from staging to shared main |
| `REJECT` | Marks node as rejected in staging (kept for audit) |
| `SWAP_SCOPE` | Toggles `project ↔ team` scope, re-prompts for final action |
| `EDIT_SUMMARY` | Updates node summary text, re-prompts for final action |
| `SKIP` | Leaves node pending |
| `QUIT` | Stops the review loop |

The interactive CLI (`cmm review`) provides a Rich-based terminal UI
as the callback. Tests use stub callbacks that return canned decisions.

All approve/reject actions are logged to the `SyncLog` for audit.

---

## CLI Reference

```
cmm init [target]             Initialize .cmm/ folder
    --shared PATH             Enable shared mode, auto-pull existing memories
    --developer NAME          Your name for sync attribution
    --team-id ID              Team identifier

cmm status [target]           Show memory stats + sync status
cmm sync [target]             Update cached_profile.md from store
cmm classify NODE_ID          Reclassify a node's scope (project|team)
    --scope project|team

cmm ingest <files...>         Ingest JSONL session files
    --project ID
    --build-profile           Build profile after ingestion
    --no-llm                  Use heuristic extraction only

cmm consolidate               Batch rebuild profiles
    --project ID | --all
    --upgrade                 Re-extract warm nodes with LLM
    --profiles-only           Skip extraction, rebuild profiles only

cmm push                      Push unpushed local nodes to shared staging
    --project ID
    --dry-run

cmm review                    Interactively review staged nodes
    --project ID
    --pending-count           Just print count, no interaction

cmm pull                      Pull approved nodes from shared store
    --project ID
    --no-team                 Skip team-scope nodes

cmm watch                     Start session watcher daemon
cmm serve                     Start MCP server (stdio transport)
cmm visualize                 Generate interactive DAG visualization
    --project ID --format html|mermaid|json|all

cmm hook start [dir]          Session-start context injection
cmm hook stop [dir]           Session-end ingestion

cmm install TARGET            Install skills to .claude/commands/
    --project ID
```

---

## Configuration

### `.cmm/config.json`

```json
{
  "auto_ingest": true,
  "auto_retrieve": true,
  "similarity_threshold": 0.85,
  "max_search_results": 5,
  "warm_extraction_only": true,
  "shared_store_path": null,
  "local_store_path": null,
  "mode": "local",
  "auto_push": false,
  "context_fill_ratio": 0.45,
  "developer_name": null,
  "team_id": null
}
```

### Environment Variables

Environment variables override config file values:

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `CMM_STORE_PATH` | Local ChromaDB directory | `~/.cognitive-memory/store` |
| `CMM_PROJECT_ID` | Default project ID | none |
| `CMM_SHARED_STORE_PATH` | Shared team store path | none (local mode) |
| `CMM_DEVELOPER_NAME` | Your name for attribution | none |
| `CMM_CONTEXT_FILL_RATIO` | Token budget ratio | `0.45` |
| `CMM_SESSION_ID` | Override session ID | derived from JSONL filename |
| `OPENAI_API_KEY` | For embedding API | none |
| `ANTHROPIC_API_KEY` | For LLM extraction | none |

---

## Module Dependency Graph

```
schemas/                         (base layer — no internal deps)
  ├── session.py
  ├── reasoning.py
  └── memory.py

ingestion/                       (depends on: schemas/)
  ├── claude_code.py
  └── watcher.py                 (also: extraction, store, compression)

extraction/                      (depends on: schemas/, anthropic)
  ├── warm_extractor.py
  └── dag_builder.py

compression/                     (depends on: schemas/, store/)
  ├── dedup.py
  └── profile_builder.py         (also: anthropic, scikit-learn)

store/                           (depends on: schemas/)
  └── vector_store.py            (also: chromadb, openai)

delivery/                        (depends on: store/)
  ├── mcp_server.py              (also: mcp)
  ├── cli_query.py               (also: evaluation/)
  └── __main__.py

discovery/                       (depends on: store/, delivery/)
  ├── project.py
  ├── hooks.py                   (also: ingestion, extraction, compression, evaluation)
  └── llms_txt.py

evaluation/                      (depends on: schemas/)
  ├── analyzer.py
  └── logger.py

sync/                            (depends on: store/, compression/)
  ├── sync.py
  └── review.py

cli.py                           (orchestrator — imports all modules)
```

---

## Test Coverage

178 tests across 16 test files:

| File | Tests | What it covers |
| ---- | ----- | -------------- |
| `test_schemas.py` | ~12 | Pydantic model validation |
| `test_parser.py` | ~8 | Claude Code JSONL parsing |
| `test_warm_extractor.py` | ~10 | Heuristic extraction strategies |
| `test_token_windowing.py` | 18 | Token budget math, windowing, overlap, dedup |
| `test_store.py` | ~8 | ChromaDB operations |
| `test_mcp_server.py` | ~6 | MCP tool responses |
| `test_watcher.py` | ~6 | Session file detection |
| `test_discovery.py` | ~8 | .cmm/ folder management |
| `test_evaluation.py` | ~6 | Session analysis and logging |
| `test_scope.py` | 11 | MemoryScope enum, serialization |
| `test_dual_store.py` | 15 | Local + shared mode, staging ops |
| `test_sync.py` | 14 | Push/pull mechanics, SyncLog |
| `test_review.py` | 11 | Approval workflow decisions |
| `test_init_shared.py` | 7 | Onboarding, classify command |
| `test_integration_e2e.py` | 3 | Full alice→carol→bob pipeline |
| `conftest.py` | — | Pytest fixtures |

### Key Constants

| Constant | Value | Purpose |
| -------- | ----- | ------- |
| `MODEL_CONTEXT_WINDOW` | 200,000 | Claude Sonnet 4.5 context size |
| `DEFAULT_FILL_RATIO` | 0.45 | Safe context utilization threshold |
| `DEFAULT_OUTPUT_RESERVE` | 4,000 | Tokens reserved for LLM response |
| `DEFAULT_OVERLAP_TOKENS` | 750 | Token overlap between adjacent windows |
| `_DEFAULT_THRESHOLD` (dedup) | 0.85 | Cosine similarity for near-duplicate |
| `_MIN_CLUSTER_SIZE` | 2 | Minimum nodes to form a cluster |
| `_MAX_NODES_PER_WINDOW` | 12 | Cap on nodes returned per LLM call |
| `_EMBED_BATCH_SIZE` | 100 | OpenAI embedding batch limit |
| `_RESOLUTION_WINDOW` | 8 | Messages window for Signal A |
| `_PIVOT_WINDOW` | 5 | Messages window for Signal C |
| `poll_interval` (watcher) | 10s | Session file polling |
| `min_file_age` (watcher) | 30s | File stability before ingestion |
