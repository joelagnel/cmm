# Fixture Format Notes

## Claude Code (JSONL)

**Location:** `~/.claude/projects/<project-slug>/<session-id>.jsonl`

**Format:** Newline-delimited JSON. Each line is a single event object.

### Event Types

#### `file-history-snapshot`
Metadata snapshot at session start. Not a message.
```json
{
  "type": "file-history-snapshot",
  "messageId": "<uuid>",
  "snapshot": { "timestamp": "...", "trackedFileBackups": {} },
  "isSnapshotUpdate": false
}
```

#### `user`
User message or tool result. `message.content` is either:
- A string (plain text user message)
- An array of content blocks

Content block types:
- `{"type": "text", "text": "..."}` — text input
- `{"type": "tool_result", "tool_use_id": "...", "content": "..."}` — tool output returned to model

```json
{
  "type": "user",
  "uuid": "<uuid>",
  "parentUuid": "<parent-uuid>",
  "sessionId": "<session-id>",
  "cwd": "/path/to/project",
  "timestamp": "2026-02-11T20:09:50.590Z",
  "message": {
    "role": "user",
    "content": "..." | [...]
  }
}
```

#### `assistant`
Assistant message. `message.content` is always an array:
- `{"type": "text", "text": "..."}` — prose response
- `{"type": "tool_use", "id": "...", "name": "Read|Write|Edit|Bash|Glob|Grep|...", "input": {...}}` — tool invocation

```json
{
  "type": "assistant",
  "uuid": "<uuid>",
  "parentUuid": "<parent-uuid>",
  "sessionId": "<session-id>",
  "timestamp": "...",
  "message": {
    "role": "assistant",
    "model": "claude-sonnet-4-5-...",
    "content": [...],
    "usage": { "input_tokens": N, "output_tokens": N, ... }
  }
}
```

### Tool Names (common)
- `Read` — read file, `input.file_path`
- `Write` — write file, `input.file_path`, `input.content`
- `Edit` — edit file, `input.file_path`
- `Bash` — shell command, `input.command`
- `Glob` — file search, `input.pattern`
- `Grep` — content search, `input.pattern`
- `TodoWrite` — task list updates

### Auto-compaction
When context gets large, Claude Code performs a compaction. This appears as a user message with a system-generated summary replacing earlier turns. Marked with `isMeta: true` in some entries.

### isSidechain
`isSidechain: true` indicates a background agent turn (e.g., a subagent spawned via the Agent tool). These are generally lower signal for reasoning extraction.

---

## Synthetic Fixtures

Located in `fixtures/synthetic/`. Hand-crafted to represent known patterns:
- `debugging_with_pivot.jsonl` — session that finds wrong root cause, pivots, succeeds
- `architectural_refactor.jsonl` — discovers unexpected coupling during refactor
- `goes_nowhere.jsonl` — session that fails without reaching a conclusion (noise test)
