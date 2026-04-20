"""Data models for tg-gemini."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Event",
    "EventType",
    "FileAttachment",
    "ImageAttachment",
    "Message",
    "ModelOption",
    "PreviewHandle",
    "ReplyContext",
]


class EventType(StrEnum):
    """Types of events from agent stream (Gemini or Claude Code)."""

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    RESULT = "result"
    ERROR = "error"
    THINKING = "thinking"
    PERMISSION_REQUEST = "permission_request"


@dataclass
class ImageAttachment:
    """An image attachment to a message."""

    mime_type: str
    data: bytes
    file_name: str = ""


@dataclass
class FileAttachment:
    """A file attachment to a message."""

    mime_type: str
    data: bytes
    file_name: str = ""


@dataclass
class Event:
    """An event from the agent stream (Gemini CLI or Claude Code)."""

    type: EventType
    content: str = ""
    tool_name: str = ""
    tool_input: str = ""
    session_id: str = ""
    done: bool = False
    error: Exception | None = None
    request_id: str = ""  # for PERMISSION_REQUEST events


@dataclass
class Message:
    """A message from a user on any platform."""

    session_key: str  # "telegram:{chatID}:{userID}"
    platform: str
    user_id: str
    user_name: str
    content: str
    message_id: str = ""
    chat_name: str = ""
    images: list[ImageAttachment] = field(default_factory=list)
    files: list[FileAttachment] = field(default_factory=list)
    reply_ctx: Any = None  # platform-specific (TG: ReplyContext dataclass)


@dataclass
class ReplyContext:
    """Telegram-specific reply context."""

    chat_id: int
    message_id: int = 0


@dataclass
class PreviewHandle:
    """Handle for an editable preview message."""

    chat_id: int
    message_id: int


@dataclass
class ModelOption:
    """A model option for the user to choose from."""

    name: str
    desc: str = ""
    alias: str = ""
