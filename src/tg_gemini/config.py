"""Configuration loading and validation for tg-gemini."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

type GeminiMode = Literal["default", "auto_edit", "yolo", "plan"]
type AgentType = Literal["gemini", "claude"]
type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type AppLanguage = Literal["", "en", "zh"]

__all__ = [
    "AgentType",
    "AppConfig",
    "AppLanguage",
    "ClaudeConfig",
    "DisplayConfig",
    "GeminiConfig",
    "GeminiMode",
    "LogConfig",
    "LogLevel",
    "RateLimitConfig",
    "SkillConfig",
    "StreamPreviewConfig",
    "TelegramConfig",
    "load_config",
    "resolve_config_path",
]

_PosInt = Annotated[int, Field(gt=0)]
_NonNegInt = Annotated[int, Field(ge=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TelegramConfig(_StrictModel):
    token: str
    allow_from: str = "*"
    group_reply_all: bool = False
    share_session_in_channel: bool = False


class GeminiConfig(_StrictModel):
    work_dir: str = "."
    model: str = ""
    mode: GeminiMode = "default"
    api_key: str = ""
    cmd: str = "gemini"
    timeout_mins: _NonNegInt = 0


class DisplayConfig(_StrictModel):
    thinking_max_len: _PosInt = 300
    tool_max_len: _PosInt = 500


class StreamPreviewConfig(_StrictModel):
    enabled: bool = True
    interval_ms: _NonNegInt = 1500
    min_delta_chars: _NonNegInt = 30
    max_chars: _NonNegInt = 2000  # 0 = no truncation


class LogConfig(_StrictModel):
    level: LogLevel = "INFO"


class RateLimitConfig(_StrictModel):
    max_messages: _NonNegInt = 0  # 0 = disabled
    window_secs: float = 60.0


class SkillConfig(_StrictModel):
    """Configuration for skill directories."""

    dirs: list[str] = Field(default_factory=list)


class ClaudeConfig(_StrictModel):
    """Configuration for Claude Code CLI agent."""

    work_dir: str = "."
    model: str = ""
    mode: str = "default"  # default | acceptEdits | plan | bypassPermissions | dontAsk
    cmd: str = "claude"
    allowed_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    timeout_mins: _NonNegInt = 0


class AppConfig(_StrictModel):
    telegram: TelegramConfig
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    agent: AgentType = "gemini"
    data_dir: str = "~/.tg-gemini"
    language: AppLanguage = ""
    log: LogConfig = Field(default_factory=LogConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    stream_preview: StreamPreviewConfig = Field(default_factory=StreamPreviewConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    skills: SkillConfig = Field(default_factory=SkillConfig)


def load_config(path: Path) -> AppConfig:
    """Load and validate configuration from a TOML file."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    cfg = AppConfig(**raw)
    logger.info(f"config loaded: {path}")
    return cfg


def resolve_config_path(explicit: str | None) -> Path:
    """Resolve config file path: explicit → local config.toml → ~/.tg-gemini/config.toml."""
    if explicit:
        return Path(explicit)
    local = Path("config.toml")
    if local.exists():
        return local
    return Path.home() / ".tg-gemini" / "config.toml"
