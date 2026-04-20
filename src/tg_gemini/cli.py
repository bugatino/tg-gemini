"""CLI entry point for tg-gemini."""

import asyncio
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

__all__ = ["app"]

app = typer.Typer(
    name="tg-gemini",
    help="Telegram↔Gemini CLI middleware service.",
    no_args_is_help=True,
)


@app.command()
def start(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config TOML file.")
    ] = None,
) -> None:
    """Start the tg-gemini bot service."""
    from tg_gemini.claude import ClaudeAgent
    from tg_gemini.config import load_config, resolve_config_path
    from tg_gemini.dedup import MessageDedup
    from tg_gemini.engine import Engine
    from tg_gemini.gemini import GeminiAgent
    from tg_gemini.i18n import I18n, Language
    from tg_gemini.ratelimit import RateLimiter
    from tg_gemini.session import SessionManager
    from tg_gemini.telegram_platform import TelegramPlatform

    config_path = resolve_config_path(str(config) if config else None)
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}", err=True)
        raise typer.Exit(1)

    cfg = load_config(config_path)

    # Configure loguru
    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.log.level,
        colorize=True,
        format="{time} {level} {message}",
    )

    _ver = version("tg-gemini")
    logger.info("tg-gemini starting", version=_ver, config=str(config_path))

    # Determine language
    lang = Language(cfg.language) if cfg.language in ("en", "zh") else Language.EN

    # Build components
    data_path = Path(cfg.data_dir).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    sessions = SessionManager(store_path=data_path / "sessions.json")

    # Create agents
    gemini_agent = GeminiAgent(
        work_dir=cfg.gemini.work_dir,
        model=cfg.gemini.model,
        mode=cfg.gemini.mode,
        cmd=cfg.gemini.cmd,
        api_key=cfg.gemini.api_key,
        timeout_mins=cfg.gemini.timeout_mins,
    )

    claude_agent = ClaudeAgent(
        work_dir=cfg.claude.work_dir,
        model=cfg.claude.model,
        mode=cfg.claude.mode,
        cmd=cfg.claude.cmd,
        allowed_tools=cfg.claude.allowed_tools or None,
        disallowed_tools=cfg.claude.disallowed_tools or None,
        timeout_mins=cfg.claude.timeout_mins,
    )

    platform = TelegramPlatform(
        token=cfg.telegram.token,
        allow_from=cfg.telegram.allow_from,
        group_reply_all=cfg.telegram.group_reply_all,
        share_session_in_channel=cfg.telegram.share_session_in_channel,
    )

    i18n = I18n(lang=lang)
    rate_limiter = RateLimiter(
        max_messages=cfg.rate_limit.max_messages, window_secs=cfg.rate_limit.window_secs
    )
    dedup = MessageDedup()

    # Extra skill dirs from config (optional; default .gemini/skills/ auto-loaded by Engine)
    extra_skill_dirs = [Path(d).expanduser() for d in cfg.skills.dirs]

    engine = Engine(
        config=cfg,
        agent=gemini_agent,
        platform=platform,
        sessions=sessions,
        i18n=i18n,
        rate_limiter=rate_limiter,
        dedup=dedup,
        skill_dirs=extra_skill_dirs,
        claude_agent=claude_agent,
    )

    async def _run() -> None:
        await rate_limiter.start()
        try:
            await engine.start()
        finally:
            await rate_limiter.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("tg-gemini stopped")
