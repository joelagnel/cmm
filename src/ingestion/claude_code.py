"""Claude Code JSONL session parser."""
import json
import re
from pathlib import Path
from datetime import datetime

from ..schemas.session import MessageRole, NormalizedSession, SessionMessage

# Tool names that indicate file modification
_WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}
# Tool names that reference files
_READ_TOOLS = {"Read", "Glob", "Grep"}
# File-path argument keys per tool
_FILE_PATH_KEYS = {"file_path", "path"}


def _extract_text(content) -> str:
    """Flatten content (str or list of blocks) to a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        inner = " ".join(
                            b.get("text", "") for b in inner if isinstance(b, dict)
                        )
                    parts.append(str(inner))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool:{block.get('name','')}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _extract_file_paths(text: str) -> list[str]:
    """Extract file paths mentioned in plain text."""
    pattern = r"(?:^|[\s'\"`(])(/[\w./\-_]+\.\w+)"
    return list({m.group(1) for m in re.finditer(pattern, text)})


def _parse_tool_use_block(block: dict) -> tuple[str, list[str], list[str]]:
    """Return (tool_name, files_referenced, files_modified) from a tool_use block."""
    name = block.get("name", "")
    inp = block.get("input", {})
    refs: list[str] = []
    mods: list[str] = []

    for key in _FILE_PATH_KEYS:
        if key in inp:
            fp = inp[key]
            if isinstance(fp, str):
                refs.append(fp)
                if name in _WRITE_TOOLS:
                    mods.append(fp)
    return name, refs, mods


class ClaudeCodeParser:
    """Parse Claude Code JSONL transcripts into NormalizedSession."""

    def parse_file(self, path: Path) -> NormalizedSession:
        """Parse a single JSONL session file."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return self._build_session(entries, path)

    def _build_session(self, entries: list[dict], source_path: Path) -> NormalizedSession:
        session_id = None
        cwd = None
        started_at = None
        ended_at = None
        messages: list[SessionMessage] = []
        raw_metadata: dict = {}

        for entry in entries:
            etype = entry.get("type", "")
            ts_str = entry.get("timestamp")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else None

            if not session_id:
                session_id = entry.get("sessionId")
            if not cwd:
                cwd = entry.get("cwd")
            if ts and not started_at:
                started_at = ts
            if ts:
                ended_at = ts

            if etype == "file-history-snapshot":
                raw_metadata["initial_snapshot"] = entry.get("snapshot", {})
                continue

            if etype == "user":
                self._handle_user(entry, ts, messages)

            elif etype == "assistant":
                self._handle_assistant(entry, ts, messages)

        if not session_id:
            session_id = source_path.stem
        if not started_at:
            started_at = datetime.now()

        return NormalizedSession(
            session_id=session_id,
            source_platform="claude_code",
            project_path=cwd,
            started_at=started_at,
            ended_at=ended_at,
            messages=messages,
            raw_metadata=raw_metadata,
        )

    def _handle_user(self, entry: dict, ts, messages: list[SessionMessage]):
        msg = entry.get("message", {})
        content = msg.get("content", "")

        # Detect isMeta (compaction summaries, slash commands)
        is_meta = entry.get("isMeta", False)
        is_sidechain = entry.get("isSidechain", False)

        if isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]

            for tr in tool_results:
                inner = tr.get("content", "")
                if isinstance(inner, list):
                    inner = "\n".join(
                        b.get("text", "") for b in inner if isinstance(b, dict) and "text" in b
                    )
                messages.append(SessionMessage(
                    role=MessageRole.TOOL_RESULT,
                    content=str(inner),
                    timestamp=ts,
                    tool_name=tr.get("tool_use_id"),
                ))

            for tb in text_blocks:
                text = tb.get("text", "")
                if text.strip():
                    messages.append(SessionMessage(
                        role=MessageRole.USER,
                        content=text,
                        timestamp=ts,
                    ))
        else:
            text = str(content)
            if text.strip() and not is_meta:
                messages.append(SessionMessage(
                    role=MessageRole.USER,
                    content=text,
                    timestamp=ts,
                ))

    def _handle_assistant(self, entry: dict, ts, messages: list[SessionMessage]):
        msg = entry.get("message", {})
        content = msg.get("content", [])

        if not isinstance(content, list):
            return

        is_sidechain = entry.get("isSidechain", False)

        for block in content:
            if not isinstance(block, dict):
                continue

            btype = block.get("type")

            if btype == "text":
                text = block.get("text", "")
                if text.strip():
                    messages.append(SessionMessage(
                        role=MessageRole.ASSISTANT,
                        content=text,
                        timestamp=ts,
                    ))

            elif btype == "tool_use":
                tool_name, refs, mods = _parse_tool_use_block(block)
                inp = block.get("input", {})
                content_str = json.dumps(inp)
                messages.append(SessionMessage(
                    role=MessageRole.TOOL_CALL,
                    content=content_str,
                    timestamp=ts,
                    tool_name=tool_name,
                    files_referenced=refs,
                    files_modified=mods,
                ))
