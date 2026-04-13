# Speaker Notes — Cognitive Memory Manager
## Applied ML Conference 2026

---

### Slide 1 — Title

**"Cognitive Memory Manager — Persistent reasoning memory for AI coding agents"**

Welcome. Today I want to talk about a problem I kept running into with Claude Code: every session starts from scratch. The agent is powerful, but it has no memory of what it learned last Tuesday. CMM is a system I built to fix that.

The three pillars: **token-budgeted extraction** (how we read sessions intelligently), **team-scale shared store** hosted on Chroma Cloud (how we get knowledge out of individual machines), and **human-gated review** (how we prevent one bad session from poisoning the whole team's knowledge).

---

### Slide 2 — The Problem: Agents Have Amnesia

Let me make the problem concrete with something I actually observed.

I was working on a deployment pipeline for a logistics platform. My agent hit a dead end — discovered that Dockerfile `CMD` using the exec form doesn't expand `$PORT`. Took 20 minutes and three failed deployments to work out. Session ended, I noted it down manually.

Three days later, same codebase, different task. The agent hit the exact same dead end. Spent another 15 minutes.

That's the third bullet: **no cross-developer transfer**. Multiply this by a team of five engineers across dozens of sessions and you're burning enormous cognitive energy rediscovering the same things. The fourth bullet — onboarding — is the same problem at hiring time. You spend your first two weeks in conversations that could have been a profile read.

---

### Slide 3 — Solution: Treat Sessions as Reasoning Traces

The key insight is that a Claude Code session transcript isn't just a chat log — it's a **reasoning trace**. It contains hypotheses that were formed, paths that failed, pivots that saved time, and solutions that actually worked.

If we can extract the structure of that reasoning and compress it into durable knowledge, we can inject it at the start of the next session. The agent doesn't start from scratch; it starts with the team's accumulated experience.

The pipeline: extract a reasoning DAG from each session, deduplicate across sessions, consolidate into a cognitive profile, and serve it back at session start.

---

### Slide 4 — End-to-End Architecture

Walk through the pipeline left to right.

**Parse** — we consume Claude Code's native JSONL format. Each session is a sequence of user/assistant events. We normalize these into a `NormalizedSession` object.

**Extract** — two tiers, covered in detail later. Either heuristic (warm) or LLM-backed (cold).

**Deduplicate** — cosine similarity at 0.85 threshold using `text-embedding-3-small`. We don't want five versions of "the database schema is in `models/`".

**Store** — ChromaDB, both local and Chroma Cloud. Vector store gives us semantic search at retrieval time.

**Profile** — agglomerative clustering over node embeddings, then an LLM pass to classify each cluster as an architectural insight, pitfall, or diagnostic strategy.

The key innovation I want to highlight is the 45% fill ratio in the cold tier — I'll explain the research behind that.

---

### Slide 5 — The Reasoning DAG

Six node types. The ones I want to draw your attention to are **PIVOT** and **DEAD_END** — these are first-class citizens in the schema. Most knowledge systems only capture what worked. We explicitly track what *didn't* work, because that's often more valuable to the next agent.

If an agent is about to try the same approach that caused a dead end two sessions ago, we can surface that dead end and tell it "this path failed, here's why." That's a very different signal from just giving it the solution.

Edges encode causality: `caused_pivot_to` means "this dead end is what caused the approach change." That lineage is what makes the DAG meaningful rather than just a bag of facts.

---

### Slide 6 — Two-Tier Extraction

The warm tier runs at every session end with zero API spend — regex heuristics, keyword scoring, explicit conclusion detection. Fast and free. Confidence is moderate (0.25–0.75).

The cold tier is LLM-backed. Claude Sonnet reads the session in token-budget windows and classifies each reasoning step. High confidence (0.50–0.95), but takes ~30 seconds and costs API credits. This is an on-demand operation — you run `cmm consolidate --upgrade` when you want to invest in quality.

The default workflow is: warm extraction happens automatically, cold extraction is triggered manually or in CI after significant sessions.

---

### Slide 7 — Token-Budget Windowing

This is the most technically interesting piece. The original implementation used fixed 5-message windows. For a long debugging session — say 80 messages — that produces 16 windows, each too small for the LLM to see any connected reasoning.

The fix: pack messages until you hit 45% of the model's context window (200K for Claude Sonnet), then start a new window with a 750-token overlap.

**Why 45%?** Three papers motivate this. The Chroma 2025 long-context benchmarks show all 18 tested frontier models degrade with length. Liu et al.'s "Lost in the Middle" paper shows 30%+ accuracy drops for information in the middle of long contexts. Paulsen's Minimum Effective Context Window paper shows effective context is much smaller than the advertised maximum. 45% is the operating point where we get large enough windows to see connected reasoning without pushing into the degradation zone.

In practice: a 6-hour session that would have produced 30 tiny windows now produces 3 large ones.

---

### Slide 8 — Knowledge Consolidation

After nodes are stored, the profile builder runs agglomerative clustering over their embeddings. Nodes that ended up in the same cluster are "about the same thing." An LLM pass then classifies each cluster:

- **Architectural insights** — structural facts. "The FastAPI app combines two services in one deployable." Doesn't change often.
- **Known pitfalls** — ranked HIGH/MEDIUM/LOW. Each gets a resolution strategy extracted from how the past agent actually solved it.
- **Diagnostic strategies** — ordered step sequences with success rates. "When deployment fails, first check X, then Y, then Z."

The profile also extracts key patterns (things the agent did that worked consistently) and anti-patterns (things that failed consistently). Both are valuable for steering future sessions.

---

### Slide 9 — Project vs. Team Scope

Every memory node carries a scope label. The examples here come directly from the supply-chain project.

The left example — offline sync using the append-only ShipmentEvent table — is completely useless outside that specific codebase. It should never cross to another team member's project.

The right example — Dockerfile `$PORT` expansion failing in exec form — applies to literally any containerized service. That should propagate to every developer who does Docker work.

The LLM at profile-build time classifies each cluster as project or team scope. Team-scope nodes flow automatically on `cmm pull` regardless of which project you're pulling into.

---

### Slide 10 — Chroma Cloud Backend

This was a recent architectural change. Originally the shared store was a second local ChromaDB at a shared filesystem path — great for a single machine, awkward for a distributed team.

We replaced the second local `PersistentClient` with Chroma Cloud's managed service. Same ChromaDB API, zero infrastructure to manage.

The local store remains a `PersistentClient` — fast, offline-first, no network dependency during a coding session. The cloud database is only hit during explicit `cmm push` and `cmm pull` operations.

Three environment variables is the entire configuration surface. The API key is **never** written to `config.json` — only read from the environment variable — so it can't accidentally get committed to a repository.

---

### Slide 11 — Push → Review → Pull

Walk through the flow.

An agent finishes a session. The warm extractor runs, producing new nodes in the local store. The developer runs `cmm push`, which copies those nodes to the staging collection in Chroma Cloud — not the main collection.

A senior developer or tech lead runs `cmm review`. They see each node with full context: type, confidence, which session it came from, which developer, the actual summary and evidence. They can approve, reject, swap scope, or edit the summary before approving.

Only approved nodes flow into the main collection. When another developer runs `cmm pull`, they get only approved nodes.

The reason for the gate: without it, a hallucinated node from a confused session becomes institutional knowledge. We saw this risk clearly when building the system — the agent occasionally produces confidently-stated but incorrect insights.

---

### Slide 12 — Human-in-the-Loop Review

The terminal UI shows the full context a reviewer needs to make a judgment call. Worth noting the example here: the original node was classified as `project` scope but the content — AWS credentials not available in Railway — is actually a team-level fact. The reviewer notices this and swaps the scope to `team` before approving.

That scope swap means when any developer on any other project runs `cmm pull`, they get that warning. One agent's discovery becomes a team-wide safeguard.

The audit trail in `sync.db` records every action with timestamp, reviewer name, and reason. This matters for regulated environments where you need to justify what knowledge the AI system was given.

---

### Slide 13 — New Developer in One Command

The onboarding flow with Chroma Cloud is:

1. Set three environment variables once (add to shell profile).
2. `cmm init . --cloud-tenant ... --cloud-database cmm --developer bob`
3. CMM initializes the `.cognitive/` folder, connects to Chroma Cloud, and immediately pulls all approved nodes — both project-specific and team-scope.
4. The `cached_profile.md` is written to `.cognitive/`, and the session-start hook is configured.

From that point on, every Claude Code session starts by reading `cached_profile.md`. The agent gets the accumulated team knowledge before it writes its first line of code.

The "tribal knowledge handoff" conversation — "oh by the way, watch out for X, and Alice tried Y last month and it failed" — becomes unnecessary.

---

### Slide 14 — Evaluation: Four Helpfulness Signals

How do we know if memory is actually helping?

**Signal A** — did memory help resolve an error? We look for the sequence: DEAD_END node → `/search-memory` invocation within a few messages → SOLUTION node within 8 messages. That's a strong causal chain.

**Signal B** — did the agent avoid a known pitfall? We embed the surfaced pitfall and check if any subsequent DEAD_END nodes have cosine similarity ≥ 0.70. If not, the pitfall appears to have been avoided. This uses embedding similarity rather than word overlap — much more robust to paraphrasing.

**Signal C** — did memory cause a useful pivot? PIVOT node within 5 messages of a `/search-memory` call suggests the retrieved information changed the agent's approach.

**Signal D** — the false-positive tracker. If a DEAD_END appears and is semantically similar to a memory that was loaded earlier in the session, the memory may have misled the agent. This is critical — we want to know when memory is causing harm, not just when it's helping.

The profile quality metrics at the bottom run locally after every ingestion, no API key needed.

---

### Slide 15 — Case Study: Iraq Logistics Simulation Platform

This is real data from the supply-chain project — an Iraq logistics simulation platform I used as the test case.

Four sessions were ingested. The profile extracted four pitfalls from those sessions, three of which are HIGH severity.

Let me walk through the most interesting ones:

**AWS credentials in Railway** — the agent deployed the app, everything looked fine locally, then production failed because AWS SDK couldn't find credentials. This is a classic "environment parity" failure. The extracted pitfall now tells every future agent: check Railway env vars for AWS credentials before deploying.

**Dockerfile `$PORT` in exec form** — this caused multiple failed deployments. The agent tried several workarounds before landing on the right fix (use `sh -c` or read the variable in Python at runtime). The extracted pitfall documents all three failed approaches plus the solution.

**`db.rollback()` in batch operations** — subtle SQLAlchemy behavior. Rolling back in an exception handler inside a loop rolls back the entire transaction, not just the failed item. This cost significant debugging time. The resolution recommends `db.begin_nested()` for savepoints.

The insights are equally valuable — particularly the offline sync architecture insight. Three different sessions touched the sync system, and all three were trying to build something that already existed. The memory prevents the fourth session from making the same mistake.

---

### Slide 16 — Results: Controlled A/B Comparison

These are actual numbers from the supply-chain project comparison.

**Deployment task** — "Deploy this project to Railway with all endpoints working, fix any errors, and get frontend communicating with backend."

- Baseline session (no memory): 3 reasoning nodes, 1 hypothesis, 0 solutions reached.
- Assisted session (memory injected): 22 reasoning nodes, 7 SOLUTION nodes. First solution at message 75.

The assisted session was structurally richer — more reasoning, more solutions identified. The memory gave the agent a starting map of the deployment architecture, which meant it spent less time exploring and more time solving.

**Profile comparison** — we compared two profiles: one built from a baseline session, one from an assisted session.

- Baseline profile: 0 architectural insights, 0 pitfalls. Just 5 key patterns and 5 anti-patterns.
- Assisted profile: 1 architectural insight (FastAPI app structure), 3 pitfalls (HIGH, HIGH, MED).

The verdict from the comparator: `assisted_richer`. The memory-assisted session produced more structured, actionable knowledge that the next session can actually use.

193 tests pass across all components with no external API keys required — embeddings and LLM calls are fully mocked in the test suite.

---

### Slide 17 — Summary

Let me close with the key points.

**The core idea** is simple: Claude Code sessions are reasoning traces. We should extract the reasoning structure and reuse it, not let it evaporate at session end.

**The technical innovations** are: token-budget windowing for quality extraction, dual-scope memory for team knowledge sharing, and Chroma Cloud as a zero-infrastructure shared backend.

**The human-in-the-loop gate** is the piece I'm most confident about. Fully automated knowledge sharing is dangerous — one confident hallucination becomes everyone's problem. The review step adds 30 seconds of human judgment and prevents institutional false knowledge.

**The results** from the supply chain case study: zero structured pitfalls without memory, three high-severity pitfalls documented with memory. That's the difference between the fourth session hitting the AWS credentials bug and knowing to check Railway env vars on day one.

Open source, MIT licensed, at github.com/sazandkhalid/cmm. Happy to take questions.

---

*Total presentation time: ~25–30 minutes at conference pace, 20 minutes if you skip the live code slides.*
