from pydantic import BaseModel, Field
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
    files_referenced: list[str] = []    # files mentioned or edited
    files_modified: list[str] = []      # files actually changed


class NormalizedSession(BaseModel):
    """Platform-agnostic session representation."""
    session_id: str
    source_platform: str                # "claude_code", "cursor", etc.
    project_path: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    messages: list[SessionMessage]
    raw_metadata: dict = Field(default_factory=dict)  # platform-specific extras
