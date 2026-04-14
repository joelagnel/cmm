"""Claude Code parser tests — Phase 1.1 exit criteria."""
import pytest
from pathlib import Path
from src.ingestion import ClaudeCodeParser
from src.schemas import MessageRole

FIXTURES = Path(__file__).parent.parent / "fixtures"
SYNTHETIC = FIXTURES / "synthetic"
REAL = FIXTURES / "claude_code"


def test_parse_synthetic_debugging_session():
    """Parse the debugging-with-pivot synthetic session."""
    parser = ClaudeCodeParser()
    session = parser.parse_file(SYNTHETIC / "debugging_with_pivot.jsonl")

    assert session.session_id == "synthetic-debug-01"
    assert session.source_platform == "claude_code"
    assert session.project_path == "/project/myapp"
    assert len(session.messages) > 5

    roles = {m.role for m in session.messages}
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles
    assert MessageRole.TOOL_CALL in roles
    assert MessageRole.TOOL_RESULT in roles


def test_parse_finds_file_modifications():
    """Parser correctly identifies files modified via Edit/Write tools."""
    parser = ClaudeCodeParser()
    session = parser.parse_file(SYNTHETIC / "debugging_with_pivot.jsonl")

    modified = [
        f
        for m in session.messages
        for f in m.files_modified
    ]
    assert any("login.py" in f for f in modified)


def test_parse_architectural_refactor():
    parser = ClaudeCodeParser()
    session = parser.parse_file(SYNTHETIC / "architectural_refactor.jsonl")

    assert session.session_id == "synthetic-arch-01"
    tool_calls = [m for m in session.messages if m.role == MessageRole.TOOL_CALL]
    assert len(tool_calls) >= 3

    # Should have Read and Glob tool calls
    tool_names = {m.tool_name for m in tool_calls}
    assert "Read" in tool_names
    assert "Glob" in tool_names


def test_parse_goes_nowhere():
    parser = ClaudeCodeParser()
    session = parser.parse_file(SYNTHETIC / "goes_nowhere.jsonl")
    assert len(session.messages) >= 4


def test_parse_real_session_detection():
    """Parse a real Claude Code session from detection project."""
    real_file = REAL / "session_detection_01.jsonl"
    if not real_file.exists():
        pytest.skip("Real fixture not available")

    parser = ClaudeCodeParser()
    session = parser.parse_file(real_file)

    assert session.session_id is not None
    assert session.source_platform == "claude_code"
    assert len(session.messages) > 10

    # Should have a mix of roles
    roles = [m.role for m in session.messages]
    assert MessageRole.ASSISTANT in roles
    assert MessageRole.TOOL_CALL in roles


def test_parse_real_session_mcp_large():
    """Parse the largest real session (MCP gateway, 1883 lines)."""
    real_file = REAL / "session_mcp_01.jsonl"
    if not real_file.exists():
        pytest.skip("Real fixture not available")

    parser = ClaudeCodeParser()
    session = parser.parse_file(real_file)

    assert len(session.messages) > 50
    # Check timestamps are ordered
    timestamps = [m.timestamp for m in session.messages if m.timestamp]
    assert timestamps == sorted(timestamps)


def test_session_timestamps_ordered():
    """Timestamps in messages should be non-decreasing."""
    parser = ClaudeCodeParser()
    session = parser.parse_file(SYNTHETIC / "debugging_with_pivot.jsonl")
    timestamps = [m.timestamp for m in session.messages if m.timestamp]
    assert timestamps == sorted(timestamps)
