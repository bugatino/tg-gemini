from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger()


class EventType(StrEnum):
    INIT = "init"
    MESSAGE = "message"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    RESULT = "result"


class BaseEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type_: EventType = Field(default=EventType.INIT, validation_alias="type")
    timestamp: datetime = Field(default_factory=datetime.now)


class InitEvent(BaseEvent):
    session_id: str
    model: str


class MessageEvent(BaseEvent):
    role: Literal["user", "assistant"]
    content: str
    delta: bool | None = None


class ToolUseEvent(BaseEvent):
    tool_name: str
    tool_id: str
    parameters: dict[str, Any]


class ToolResultError(BaseModel):
    message: str


class ToolResultEvent(BaseEvent):
    tool_id: str
    status: Literal["success", "error"]
    output: str | None = None
    error: ToolResultError | None = None


class ErrorEvent(BaseEvent):
    severity: Literal["warning", "error"]
    message: str


class ModelStats(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cached: int
    _input: int


class StreamStats(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cached: int
    _input: int
    duration_ms: int
    tool_calls: int
    models: dict[str, ModelStats]


class ResultEvent(BaseEvent):
    status: Literal["success", "error"]
    error: dict[str, Any] | None = None
    stats: StreamStats | None = None


GeminiEvent = (
    InitEvent | MessageEvent | ToolUseEvent | ToolResultEvent | ErrorEvent | ResultEvent
)


def parse_event(data: dict[str, Any]) -> GeminiEvent:
    event_type = data.get("type")
    logger.debug("parsing_event", event_type=event_type)
    if event_type == EventType.INIT:
        return InitEvent.model_validate(data)
    if event_type == EventType.MESSAGE:
        return MessageEvent.model_validate(data)
    if event_type == EventType.TOOL_USE:
        return ToolUseEvent.model_validate(data)
    if event_type == EventType.TOOL_RESULT:
        return ToolResultEvent.model_validate(data)
    if event_type == EventType.ERROR:
        return ErrorEvent.model_validate(data)
    if event_type == EventType.RESULT:
        return ResultEvent.model_validate(data)
    msg = f"Unknown event type: {event_type}"
    raise ValueError(msg)
