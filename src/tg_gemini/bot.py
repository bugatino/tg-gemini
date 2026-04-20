from __future__ import annotations

import asyncio
import contextlib
import html as html_mod
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender

if TYPE_CHECKING:
    from tg_gemini.config import AppConfig

from tg_gemini.events import (
    ErrorEvent,
    InitEvent,
    MessageEvent,
    ResultEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from tg_gemini.gemini import GeminiAgent, SessionInfo
from tg_gemini.markdown import markdown_to_html, split_message
from tg_gemini.sessions import PersistedSession, SessionStore

logger = structlog.get_logger()

TELEGRAM_MAX_LENGTH = 4096
UPDATE_INTERVAL = 1.5
UPDATE_CHAR_THRESHOLD = 200
TOOL_CMD_TRUNCATE = 4096
TOOL_PARAM_TRUNCATE = 4096
TOOL_CONTENT_PREVIEW = 500


@dataclass
class UserSession:
    session_id: str | None = None
    model: str | None = None
    active: bool = True
    # Guards mutations of session_id, model, custom_names.
    # Held for microseconds, NOT for the entire streaming duration.
    mutation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_sessions: list[SessionInfo] = field(default_factory=list)
    custom_names: dict[str, str] = field(default_factory=dict)
    pending_name: str | None = None
    # Set by the stop button to abort the current in-flight stream.
    # Not persisted — only meaningful during an active stream.
    stop_event: asyncio.Event | None = None


class SessionManager:
    """Manages per-user sessions, backed by an optional SessionStore for persistence."""

    def __init__(self, store: SessionStore | None = None) -> None:
        self._store = store
        self._sessions: dict[int, UserSession] = {}

    @classmethod
    async def create(cls, store: SessionStore) -> SessionManager:
        """Factory: loads persisted sessions from disk and returns a ready manager."""
        self = cls(store)
        persisted = await store.load()
        for uid, data in persisted.items():
            self._sessions[uid] = UserSession(
                session_id=data.session_id,
                model=data.model,
                custom_names=data.custom_names,
            )
        logger.info("sessions_restored", count=len(self._sessions))
        return self

    def get(self, user_id: int) -> UserSession:
        if user_id not in self._sessions:
            logger.debug("session_created", user_id=user_id)
            self._sessions[user_id] = UserSession()
        else:
            logger.debug("session_retrieved", user_id=user_id)
        return self._sessions[user_id]

    async def save(self, user_id: int) -> None:
        """Persist one user's session state to disk after mutations."""
        if self._store is None:
            return
        session = self._sessions.get(user_id)
        if session is None:
            return
        await self._store.save(
            user_id,
            PersistedSession(
                session_id=session.session_id,
                model=session.model,
                custom_names=session.custom_names,
            ),
        )

    async def shutdown(self) -> None:
        """Persist all sessions to disk and release resources. Call on SIGTERM/SIGINT."""
        if self._store is None:
            return
        all_sessions = {
            uid: PersistedSession(
                session_id=s.session_id, model=s.model, custom_names=s.custom_names
            )
            for uid, s in self._sessions.items()
        }
        await self._store.save_all(all_sessions)
        logger.info("sessions_persisted_on_shutdown", count=len(all_sessions))


def _parse_allowed_ids(allow_from: str) -> list[int]:
    """Parse allow_from string (e.g., '*' or '123,456') into list of allowed user IDs."""
    if allow_from == "*":
        return []
    return [int(x.strip()) for x in allow_from.split(",") if x.strip().isdigit()]


def _is_authorized(user_id: int, allowed_ids: list[int]) -> bool:
    return not allowed_ids or user_id in allowed_ids


router = Router()

# ── Inline keyboard builders ────────────────────────────────────────────────────


def _build_model_keyboard() -> InlineKeyboardMarkup:
    """Build a one-row keyboard with all available model aliases."""
    models = ["auto", "pro", "flash", "flash-lite"]
    row = [InlineKeyboardButton(text=m, callback_data=f"m:{m}") for m in models]
    return InlineKeyboardMarkup(inline_keyboard=[row])


def _build_session_keyboard(
    sessions: list[SessionInfo], active_id: str | None, custom_names: dict[str, str]
) -> InlineKeyboardMarkup | None:
    """Build a keyboard where each session is a row with Resume + Delete buttons.

    Returns None if the session list is empty.
    """
    if not sessions:
        return None

    rows: list[list[InlineKeyboardButton]] = []
    for s in sessions:
        title = custom_names.get(s.session_id, s.title)
        # Truncate long titles so they don't blow out Telegram's 64-char callback limit
        display = title[:30] + "…" if len(title) > 30 else title
        time_str = f" ({s.time})" if s.time else ""

        if s.session_id == active_id:
            # Active session: Resume becomes a non-clickable "Current" label
            resume_btn = InlineKeyboardButton(
                text="Current", callback_data="noop:current"
            )
        else:
            resume_btn = InlineKeyboardButton(
                text="Resume", callback_data=f"r:{s.session_id}"
            )

        delete_btn = InlineKeyboardButton(
            text="Delete", callback_data=f"d:{s.session_id}"
        )

        rows.append(
            [
                InlineKeyboardButton(
                    text=f"📄 {display}{time_str}", callback_data="noop:info"
                ),
                resume_btn,
                delete_btn,
            ]
        )

    return InlineKeyboardMarkup(inline_keyboard=rows)


def _build_stop_button(reply_msg_id: int) -> InlineKeyboardMarkup:
    """Build a single stop button for an in-progress stream."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⏹ Stop", callback_data=f"s:{reply_msg_id}")]
        ]
    )


# ── Callback query handlers ─────────────────────────────────────────────────────


@router.callback_query(F.data.startswith("m:"))
async def callback_model(query: CallbackQuery, sessions: SessionManager) -> None:
    """Handle model selection from the inline keyboard."""
    if not query.message or not query.from_user or not query.data:
        return

    model_name = query.data[2:]  # strip "m:" prefix
    session = sessions.get(query.from_user.id)

    async with session.mutation_lock:
        session.model = model_name

    await sessions.save(query.from_user.id)
    logger.info(
        "model_selected_via_keyboard", user_id=query.from_user.id, model=model_name
    )

    assert isinstance(query.message, Message)
    with contextlib.suppress(Exception):
        await query.message.edit_text(
            f"✅ Model set to: <b>{model_name}</b>", parse_mode="HTML"
        )
    await query.answer()


@router.callback_query(F.data.startswith("r:"))
async def callback_resume(query: CallbackQuery, sessions: SessionManager) -> None:
    """Handle session resume from the inline keyboard."""
    if not query.message or not query.from_user or not query.data:
        return

    session_id = query.data[2:]  # strip "r:" prefix
    session = sessions.get(query.from_user.id)

    async with session.mutation_lock:
        session.session_id = session_id

    await sessions.save(query.from_user.id)
    logger.info(
        "session_resumed_via_keyboard",
        user_id=query.from_user.id,
        session_id=session_id,
    )

    assert isinstance(query.message, Message)
    with contextlib.suppress(Exception):
        await query.message.edit_text(
            f"✅ Resuming session: <code>{session_id}</code>", parse_mode="HTML"
        )
    await query.answer()


@router.callback_query(F.data.startswith("d:"))
async def callback_delete(
    query: CallbackQuery, sessions: SessionManager, agent: GeminiAgent
) -> None:
    """Handle session deletion from the inline keyboard."""
    if not query.message or not query.from_user or not query.data:
        return

    target_id = query.data[2:]  # strip "d:" prefix
    session = sessions.get(query.from_user.id)

    success = await agent.delete_session(target_id)
    if success:
        async with session.mutation_lock:
            if session.session_id == target_id:
                session.session_id = None
            session.custom_names.pop(target_id, None)
        await sessions.save(query.from_user.id)
        logger.info(
            "session_deleted_via_keyboard",
            user_id=query.from_user.id,
            session_id=target_id,
        )
        assert isinstance(query.message, Message)
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                f"🗑 Deleted session: <code>{target_id}</code>", parse_mode="HTML"
            )
    else:
        assert isinstance(query.message, Message)
        with contextlib.suppress(Exception):
            await query.message.edit_text("❌ Failed to delete session.")
    await query.answer()


@router.callback_query(F.data.startswith("s:"))
async def callback_stop(query: CallbackQuery, sessions: SessionManager) -> None:
    """Handle stop button press during an active stream."""
    if not query.from_user or not query.data:
        return

    try:
        _ = int(query.data[2:])  # message_id, for future reference
    except ValueError:
        await query.answer("Already stopped.", show_alert=True)
        return

    session = sessions.get(query.from_user.id)
    if session.stop_event is not None:
        session.stop_event.set()
        logger.info("stream_stop_requested", user_id=query.from_user.id)

    await query.answer("Stopping…")


@router.callback_query(F.data.startswith("noop:"))
async def callback_noop(query: CallbackQuery) -> None:
    """Acknowledge non-action button presses (e.g., session title labels)."""
    await query.answer()  # dismiss loading indicator, no other action


@router.message(Command("start"))
async def cmd_start(message: Message, sessions: SessionManager) -> None:
    if not message.from_user:
        return
    logger.info("command_start", user_id=message.from_user.id)
    session = sessions.get(message.from_user.id)
    session.active = True
    await message.answer(
        "Bot activated. Send me a message to chat with Gemini.\n\n"
        "Commands:\n"
        "/new [name] - Start a new session (with optional name)\n"
        "/list - List available sessions\n"
        "/resume [index|id] - Resume a specific or the latest session\n"
        "/name <name> - Name the current session\n"
        "/delete <index|id> - Delete a session\n"
        "/model <name> - Switch model\n"
        "/status - Show current status\n"
        "/current - Show current status (detailed)"
    )


@router.message(Command("new"))
async def cmd_new(
    message: Message, command: CommandObject, sessions: SessionManager
) -> None:
    if not message.from_user:
        return
    logger.info("command_new", user_id=message.from_user.id, args=command.args)
    session = sessions.get(message.from_user.id)
    async with session.mutation_lock:
        session.session_id = None
        session.pending_name = command.args or None
    await sessions.save(message.from_user.id)

    msg = "Session cleared. Next message starts a new conversation"
    if session.pending_name:
        msg += f" named: {session.pending_name}"
    await message.answer(msg + ".")


async def _format_session_list(
    sessions: list[SessionInfo], active_id: str | None, custom_names: dict[str, str]
) -> str:
    if not sessions:
        return "No sessions found."
    lines = ["Available sessions:"]
    for s in sessions:
        marker = "▶" if s.session_id == active_id else "◻"
        title = custom_names.get(s.session_id, s.title)
        lines.append(f"{s.index}. {marker} {title} ({s.time})")
    lines.append("\nUse `/resume <index>` to change session.")
    return "\n".join(lines)


@router.message(Command("list"))
async def cmd_list(
    message: Message, sessions: SessionManager, agent: GeminiAgent
) -> None:
    if not message.from_user:
        return
    logger.info("command_list", user_id=message.from_user.id)
    session = sessions.get(message.from_user.id)
    session.last_sessions = await agent.list_sessions()

    keyboard = _build_session_keyboard(
        session.last_sessions, session.session_id, session.custom_names
    )

    if keyboard is None:
        await message.answer("No sessions found.")
        return

    await message.answer("Your sessions — tap Resume or Delete:", reply_markup=keyboard)


def _resolve_id(arg: str, last_sessions: list[SessionInfo]) -> str:
    if arg.isdigit():
        idx = int(arg)
        for s in last_sessions:
            if s.index == idx:
                return s.session_id
    return arg


@router.message(Command("name"))
async def cmd_name(
    message: Message, command: CommandObject, sessions: SessionManager
) -> None:
    if not message.from_user:
        return
    logger.info("command_name", user_id=message.from_user.id, args=command.args)
    session = sessions.get(message.from_user.id)
    if not session.session_id:
        await message.answer("No active session to name. Send a message first.")
        return
    if not command.args:
        await message.answer("Usage: /name <new_name>")
        return
    async with session.mutation_lock:
        session.custom_names[session.session_id] = command.args
    await sessions.save(message.from_user.id)
    await message.answer(f"Session renamed to: {command.args}")


@router.message(Command("resume"))
async def cmd_resume(
    message: Message, command: CommandObject, sessions: SessionManager
) -> None:
    if not message.from_user:
        return
    logger.info("command_resume", user_id=message.from_user.id, args=command.args)
    session = sessions.get(message.from_user.id)
    if command.args:
        target_id = _resolve_id(command.args, session.last_sessions)
        async with session.mutation_lock:
            session.session_id = target_id
        await sessions.save(message.from_user.id)
        await message.answer(
            f"Resuming session: <code>{target_id}</code>", parse_mode="HTML"
        )
    else:
        async with session.mutation_lock:
            session.session_id = "latest"
        await sessions.save(message.from_user.id)
        await message.answer("Resuming latest session.")


@router.message(Command("delete"))
async def cmd_delete(
    message: Message,
    command: CommandObject,
    sessions: SessionManager,
    agent: GeminiAgent,
) -> None:
    if not message.from_user:
        return
    logger.info("command_delete", user_id=message.from_user.id, args=command.args)
    if not command.args:
        await message.answer("Usage: /delete <index|id>")
        return

    session = sessions.get(message.from_user.id)
    target_id = _resolve_id(command.args, session.last_sessions)

    success = await agent.delete_session(target_id)
    if success:
        async with session.mutation_lock:
            if session.session_id == target_id:
                session.session_id = None
            session.custom_names.pop(target_id, None)
        await sessions.save(message.from_user.id)
        await message.answer(
            f"Deleted session: <code>{target_id}</code>", parse_mode="HTML"
        )
    else:
        await message.answer("Failed to delete session.")


@router.message(Command("model"))
async def cmd_model(
    message: Message, command: CommandObject, sessions: SessionManager
) -> None:
    if not message.from_user:
        return
    logger.info("command_model", user_id=message.from_user.id, args=command.args)
    if not command.args:
        await message.answer("Select a model:", reply_markup=_build_model_keyboard())
        return
    session = sessions.get(message.from_user.id)
    async with session.mutation_lock:
        session.model = command.args
    await sessions.save(message.from_user.id)
    await message.answer(f"Model set to: {command.args}")


@router.message(Command("status"))
async def cmd_status(
    message: Message, sessions: SessionManager, config: AppConfig
) -> None:
    if not message.from_user:
        return
    logger.info("command_status", user_id=message.from_user.id)
    session = sessions.get(message.from_user.id)
    status = (
        f"Active: {session.active}\n"
        f"Model: {session.model or config.gemini.model}\n"
        f"Session: {session.session_id or 'new'}"
    )
    await message.answer(status)


@router.message(Command("current"))
async def cmd_current(
    message: Message, sessions: SessionManager, config: AppConfig
) -> None:
    if not message.from_user:
        return
    logger.info("command_current", user_id=message.from_user.id)
    session = sessions.get(message.from_user.id)
    current = (
        f"Current Model: <code>{session.model or config.gemini.model}</code>\n"
        f"Current Session: <code>{session.session_id or 'new'}</code>"
    )
    await message.answer(current, parse_mode="HTML")


def _truncate(text: str, limit: int) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def _esc(text: str) -> str:
    return html_mod.escape(text)


def _pre(text: str, lang: str = "") -> str:
    cls = f' class="language-{lang}"' if lang else ""
    return f"<pre><code{cls}>{_esc(text)}</code></pre>"


def _fmt_shell(params: dict[str, object]) -> str:
    cmd = _truncate(str(params["command"]), TOOL_CMD_TRUNCATE)
    desc = params.get("description", "")
    title = _esc(str(desc)) if desc else "run_shell_command"
    return f"🔧 <b>{title}</b>\n{_pre(cmd, 'bash')}"


def _fmt_file_op(name: str, params: dict[str, object]) -> str:
    fp = _esc(str(params["file_path"]))
    parts: list[str] = [f"🔧 <b>{name}</b>: <code>{fp}</code>"]
    if name == "replace":
        if "instruction" in params:
            instr = _esc(_truncate(str(params["instruction"]), TOOL_PARAM_TRUNCATE))
            parts.append(f"<i>{instr}</i>")
        old = str(params.get("old_string", ""))
        new = str(params.get("new_string", ""))
        if old or new:
            diff_lines = []
            if old:
                diff_lines.append(f"- {_truncate(old, TOOL_CONTENT_PREVIEW)}")
            if new:
                diff_lines.append(f"+ {_truncate(new, TOOL_CONTENT_PREVIEW)}")
            parts.append(_pre("\n".join(diff_lines)))
    elif name == "write_file" and "content" in params:
        preview = _truncate(str(params["content"]), TOOL_CONTENT_PREVIEW)
        parts.append(_pre(preview))
    elif name == "read_file":
        start = params.get("start_line")
        end = params.get("end_line")
        if start and end:
            parts[0] += f" (L{start}-L{end})"
        elif start:
            parts[0] += f" (from L{start})"
    return "\n".join(parts)


def _fmt_search(name: str, params: dict[str, object]) -> str | None:
    if name == "list_directory" and "dir_path" in params:
        return f"🔧 <b>list_directory</b>: <code>{_esc(str(params['dir_path']))}</code>"
    if name == "glob" and "pattern" in params:
        return f"🔧 <b>glob</b>: <code>{_esc(str(params['pattern']))}</code>"
    if name == "grep_search" and ("pattern" in params or "query" in params):
        query = str(params.get("pattern") or params.get("query", ""))
        return f"🔧 <b>grep_search</b>: <code>{_esc(query)}</code>"
    if name == "google_web_search" and "query" in params:
        return f"🔧 <b>google_web_search</b>: {_esc(str(params['query']))}"
    if name == "web_fetch" and ("prompt" in params or "url" in params):
        val = str(params.get("prompt") or params.get("url", ""))
        return f"🔧 <b>web_fetch</b>: {_esc(_truncate(val, TOOL_PARAM_TRUNCATE))}"
    return None


def _format_tool_html(event: ToolUseEvent) -> str:
    """Format a tool use event into HTML for Telegram display."""
    name = event.tool_name
    params = event.parameters

    if name == "run_shell_command" and "command" in params:
        return _fmt_shell(params)
    if name in ("read_file", "write_file", "replace") and "file_path" in params:
        return _fmt_file_op(name, params)

    result = _fmt_search(name, params)
    if result:
        return result

    if params:
        first_val = _truncate(str(next(iter(params.values()))), TOOL_PARAM_TRUNCATE)
        return f"🔧 <b>{name}</b>: {_esc(first_val)}"

    return f"🔧 {name}"


async def _throttle_edit(
    reply: Message, accumulated: str, last_update_time: float, last_update_len: int
) -> tuple[float, int]:
    now = time.monotonic()
    if (
        now - last_update_time >= UPDATE_INTERVAL
        or len(accumulated) - last_update_len >= UPDATE_CHAR_THRESHOLD
    ):
        await _edit_reply(reply, accumulated)
        return now, len(accumulated)
    return last_update_time, last_update_len


@dataclass
class _StreamState:
    accumulated: str = ""
    tool_messages: dict[str, Message] = field(default_factory=dict)
    tool_html: dict[str, str] = field(default_factory=dict)
    last_update_time: float = 0.0
    last_update_len: int = 0
    aborted: bool = False
    stats_footer: str = ""


async def _handle_event(
    event: object, session: UserSession, state: _StreamState, reply: Message
) -> None:
    """Process a single stream event, updating state and UI."""
    if isinstance(event, InitEvent):
        logger.debug("stream_init", session_id=event.session_id, model=event.model)
        async with session.mutation_lock:
            session.session_id = event.session_id
            if session.pending_name and event.session_id:
                session.custom_names[event.session_id] = session.pending_name
                session.pending_name = None
        # Inject the stop button so the user can abort this stream
        if reply.chat and reply.message_id:
            with contextlib.suppress(Exception):
                await reply.edit_text(
                    "Thinking…", reply_markup=_build_stop_button(reply.message_id)
                )
    elif isinstance(event, MessageEvent) and event.role == "assistant":
        state.accumulated = (
            (state.accumulated + event.content) if event.delta else event.content
        )
        state.last_update_time, state.last_update_len = await _throttle_edit(
            reply, state.accumulated, state.last_update_time, state.last_update_len
        )
    elif isinstance(event, ToolUseEvent):
        logger.debug(
            "stream_tool_use", tool_name=event.tool_name, tool_id=event.tool_id
        )
        tool_html = _format_tool_html(event)
        tool_msg = await reply.answer(tool_html, parse_mode="HTML")
        state.tool_messages[event.tool_id] = tool_msg
        state.tool_html[event.tool_id] = tool_html
    elif isinstance(event, ToolResultEvent) and event.tool_id in state.tool_messages:
        logger.debug("stream_tool_result", tool_id=event.tool_id, status=event.status)
        tool_msg = state.tool_messages[event.tool_id]
        icon = "✅" if event.status == "success" else "❌"
        new_html = state.tool_html[event.tool_id].replace("🔧", icon, 1)
        with contextlib.suppress(Exception):
            await tool_msg.edit_text(new_html, parse_mode="HTML")
    elif isinstance(event, ErrorEvent):
        logger.error("stream_error", severity=event.severity, message=event.message)
        await reply.edit_text(f"Error: {event.message}", reply_markup=None)
        state.aborted = True
    elif isinstance(event, ResultEvent) and event.stats:
        s = event.stats
        state.stats_footer = f"({s.total_tokens} tokens, {s.duration_ms / 1000:.1f}s)"
        logger.info(
            "stream_result",
            status=event.status,
            total_tokens=s.total_tokens,
            duration_s=s.duration_ms / 1000,
        )


async def _process_stream(
    message: Message,
    session: UserSession,
    agent: GeminiAgent,
    session_id: str | None,
    model: str | None,
    _sessions: SessionManager,
) -> tuple[str, list[str]]:
    if not message.bot:
        return "", []

    logger.info(
        "stream_started", user_id=message.from_user.id if message.from_user else None
    )
    reply = await message.answer("Thinking…")
    state = _StreamState(last_update_time=time.monotonic())

    # Wire up stop signal so the /stop button can abort this stream
    stop_evt = asyncio.Event()
    session.stop_event = stop_evt

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        async for event in agent.run_stream(
            message.text or "", session_id, model, stop_event=stop_evt
        ):
            await _handle_event(event, session, state, reply)
            if state.aborted:
                break

    # Clean up stop event reference
    session.stop_event = None

    if state.aborted:
        # Error or stop already handled by _handle_event — don't overwrite
        pass
    elif state.accumulated:
        if state.tool_messages:
            # Tools were used: delete "Thinking..." and send response as new message
            # so it appears AFTER tool messages in correct order
            with contextlib.suppress(Exception):
                await reply.delete()
            logger.debug(
                "stream_ending_with_tools", num_tool_msgs=len(state.tool_messages)
            )
            await _send_new(reply, state.accumulated)
        else:
            # No tools: edit "Thinking..." in place (clean Q&A flow)
            logger.debug("stream_ending_no_tools", response_len=len(state.accumulated))
            await _send_final(reply, state.accumulated)
    elif not state.tool_messages:
        logger.debug("stream_no_response")
        await reply.edit_text("No response.", reply_markup=None)

    if state.stats_footer:
        with contextlib.suppress(Exception):
            await reply.answer(f"<i>{state.stats_footer}</i>", parse_mode="HTML")

    return state.accumulated, list(state.tool_messages.keys())


@router.message(F.text & ~F.text.startswith("/"))
async def handle_message(
    message: Message, sessions: SessionManager, agent: GeminiAgent, config: AppConfig
) -> None:
    if not message.from_user or not message.text:
        return
    if not _is_authorized(
        message.from_user.id, _parse_allowed_ids(config.telegram.allow_from)
    ):
        return

    session = sessions.get(message.from_user.id)
    if not session.active:
        return

    # Capture session state BEFORE streaming. These locals remain stable even if
    # concurrent commands mutate the session object during the stream.
    session_id = session.session_id
    model = session.model

    # Stream WITHOUT holding any lock — commands remain unblocked
    await _process_stream(message, session, agent, session_id, model, sessions)

    # Persist session state after the stream completes
    await sessions.save(message.from_user.id)


async def _edit_reply(reply: Message, accumulated: str) -> None:
    """Edit the reply message with current accumulated text (streaming preview)."""
    html = markdown_to_html(accumulated)
    chunks = split_message(html, max_len=TELEGRAM_MAX_LENGTH)
    if chunks:
        with contextlib.suppress(Exception):
            await reply.edit_text(chunks[0], parse_mode="HTML")


async def _send_final(reply: Message, accumulated: str) -> None:
    """Edit the reply with final response, splitting into extra messages if needed."""
    html = markdown_to_html(accumulated)
    chunks = split_message(html, max_len=TELEGRAM_MAX_LENGTH)

    if not chunks:
        return

    with contextlib.suppress(Exception):
        await reply.edit_text(chunks[0], parse_mode="HTML", reply_markup=None)

    if reply.bot:
        for chunk in chunks[1:]:
            with contextlib.suppress(Exception):
                await reply.answer(chunk, parse_mode="HTML")


async def _send_new(reply: Message, accumulated: str) -> None:
    """Send the final response as new message(s) after tool messages."""
    html = markdown_to_html(accumulated)
    chunks = split_message(html, max_len=TELEGRAM_MAX_LENGTH)

    for chunk in chunks:
        with contextlib.suppress(Exception):
            await reply.answer(chunk, parse_mode="HTML")


async def start_bot(config: AppConfig) -> None:
    bot = Bot(token=config.telegram.token)

    commands = [
        BotCommand(command="start", description="Welcome and help"),
        BotCommand(command="new", description="Start a new session"),
        BotCommand(command="list", description="List your sessions"),
        BotCommand(command="resume", description="Resume a session"),
        BotCommand(command="name", description="Name the current session"),
        BotCommand(command="current", description="Show current status"),
        BotCommand(command="model", description="Switch Gemini model"),
        BotCommand(command="delete", description="Delete a session"),
    ]
    await bot.set_my_commands(commands)

    dp = Dispatcher()
    dp.include_router(router)

    # Create store and load persisted sessions
    sessions_path = Path.home() / ".config" / "tg-gemini" / "sessions.json"
    store = SessionStore(_path=sessions_path)
    sessions = await SessionManager.create(store)

    agent = GeminiAgent(
        work_dir=config.gemini.work_dir,
        model=config.gemini.model,
        mode=config.gemini.mode,
        cmd=config.gemini.cmd,
        api_key=config.gemini.api_key,
        timeout_mins=config.gemini.timeout_mins,
    )

    # Register graceful shutdown to persist sessions
    shutdown_requested = asyncio.Event()

    async def _shutdown() -> None:
        logger.info("shutdown_signal_received")
        await sessions.shutdown()
        await bot.session.close()
        shutdown_requested.set()

    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown()))
    except NotImplementedError:
        # Windows doesn't support add_signal_handler; rely on KeyboardInterrupt
        pass

    logger.info("bot_polling_started")
    try:
        polling_task = asyncio.create_task(
            dp.start_polling(bot, sessions=sessions, agent=agent, config=config)
        )
        await asyncio.gather(polling_task, shutdown_requested.wait())
    finally:
        # Final fallback persist (in case polling exits without a signal)
        await sessions.shutdown()
