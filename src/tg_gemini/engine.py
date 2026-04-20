"""Engine: orchestrates Telegram messages → agent sessions → streaming replies."""

import asyncio
import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from tg_gemini.card import Card, CardBuilder, CardButton
from tg_gemini.claude import ClaudeAgent, ClaudeSession
from tg_gemini.commands import CommandLoader
from tg_gemini.config import AgentType, AppConfig
from tg_gemini.dedup import MessageDedup
from tg_gemini.gemini import GeminiAgent, GeminiSession, _FALLBACK_MODELS, _is_quota_error
from tg_gemini.i18n import I18n, Language, MsgKey
from tg_gemini.models import EventType, Message, ReplyContext
from tg_gemini.ratelimit import RateLimiter
from tg_gemini.session import Session, SessionManager
from tg_gemini.skills import SkillRegistry
from tg_gemini.streaming import StreamPreview
from tg_gemini.telegram_platform import TelegramPlatform

__all__ = ["Engine"]

_MAX_QUEUE = 5
_PAGE_SIZE = 5
_HISTORY_DISPLAY = 10  # recent history entries shown by /history


def _to_tg_command(name: str) -> str:
    """Normalize an arbitrary name to a valid Telegram command slug.

    Rules: lowercase, only [a-z0-9_], collapse runs of underscores,
    strip leading/trailing underscores, truncate to 32 chars.
    """
    slug = re.sub(r"[^a-z0-9]", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:32] or "cmd"


@dataclass
class _InteractiveState:
    quiet: bool = False
    show_tool_output: bool = False
    delete_selected: set[str] = field(default_factory=set)
    delete_phase: str = ""  # "" | "select" | "confirm"


@dataclass
class _PendingPermission:
    """A pending permission request awaiting user response."""

    request_id: str
    tool_name: str
    tool_input: str
    ctx: ReplyContext


class Engine:
    """Routes incoming messages to Gemini or Claude and streams responses back to Telegram."""

    def __init__(
        self,
        config: AppConfig,
        agent: GeminiAgent,
        platform: TelegramPlatform,
        sessions: SessionManager,
        i18n: I18n,
        rate_limiter: RateLimiter | None = None,
        dedup: MessageDedup | None = None,
        skill_dirs: list[Path] | None = None,
        claude_agent: ClaudeAgent | None = None,
    ) -> None:
        self._config = config
        self._gemini = agent
        self._claude = claude_agent
        self._platform = platform
        self._sessions = sessions
        self._i18n = i18n
        self._rate_limiter = rate_limiter or RateLimiter()
        self._dedup = dedup or MessageDedup()
        self._share_session = config.telegram.share_session_in_channel
        self._queues: dict[str, asyncio.Queue[Message]] = {}
        self._active_gemini: dict[str, GeminiSession] = {}
        self._active_claude: dict[str, ClaudeSession] = {}
        self._interactive: dict[str, _InteractiveState] = {}
        self._pending_permissions: dict[str, _PendingPermission] = {}

        # Load Commands and Skills (auto-load from .gemini/commands/ and .gemini/skills/)
        work_dir = Path(config.gemini.work_dir).expanduser().resolve()
        logger.info(f"work_dir: {work_dir}")
        self._cmd_loader = CommandLoader(work_dir)
        cmd_count = self._cmd_loader.load()
        self._skill_registry = SkillRegistry(work_dir)
        # Add extra skill dirs from config
        for d in skill_dirs or []:
            self._skill_registry.add_directory(d)
        skill_count = self._skill_registry.load()
        logger.info(
            f"commands and skills loaded: {cmd_count} commands, {skill_count} skills"
        )

    async def start(self) -> None:
        """Start the Telegram polling loop."""
        self._platform.register_callback_prefix("cmd:", self._handle_cmd_callback)
        self._platform.register_callback_prefix("act:", self._handle_act_callback)
        self._platform.register_callback_prefix("sel:", self._handle_sel_callback)
        self._platform.register_callback_prefix("perm:", self._handle_perm_callback)
        await self._platform.start(
            self.handle_message, on_started=self._refresh_commands_menu
        )

    async def stop(self) -> None:
        """Stop the platform."""
        await self._platform.stop()

    async def handle_message(self, msg: Message) -> None:
        """Entry point for all incoming messages."""
        logger.info(
            "message received",
            platform=msg.platform,
            user=msg.user_name,
            content_len=len(msg.content),
        )

        content = msg.content.strip()
        if not content and not msg.images and not msg.files:
            return

        # Dedup check
        if self._dedup.is_duplicate(msg.message_id):
            logger.debug("Engine: duplicate message ignored", message_id=msg.message_id)
            return

        # Rate limit check
        if not self._rate_limiter.allow(msg.session_key):
            await self._reply(msg, self._i18n.t(MsgKey.RATE_LIMITED))
            return

        # Slash command handling
        if content.startswith("/"):
            await self.handle_command(msg, content)
            return

        session = self._sessions.get_or_create(msg.session_key)

        # If agent is busy, queue the message
        if session.busy:
            q = self._queues.setdefault(
                msg.session_key, asyncio.Queue(maxsize=_MAX_QUEUE)
            )
            if q.full():
                await self._reply(msg, self._i18n.t(MsgKey.SESSION_BUSY))
                return
            await q.put(msg)
            await self._reply(msg, self._i18n.t(MsgKey.SESSION_BUSY))
            return

        await self._process(msg, session)

    async def handle_command(self, msg: Message, raw: str) -> bool:
        """Handle slash commands. Returns True if consumed."""
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]  # strip @botname suffix
        args = parts[1].strip() if len(parts) > 1 else ""

        match cmd:
            case "/new":
                await self._cmd_new(msg)
            case "/help":
                await self._cmd_help(msg)
            case "/stop":
                await self._cmd_stop(msg)
            case "/model":
                await self._cmd_model(msg, args)
            case "/mode":
                await self._cmd_mode(msg, args)
            case "/agent":
                await self._cmd_agent(msg, args)
            case "/lang":
                await self._cmd_lang(msg, args)
            case "/quiet":
                await self._cmd_quiet(msg)
            case "/toolout":
                await self._cmd_toolout(msg)
            case "/status":
                await self._cmd_status(msg)
            case "/list":
                await self._cmd_list(msg, args)
            case "/current":
                await self._cmd_current(msg)
            case "/history":
                await self._cmd_history(msg)
            case "/switch":
                await self._cmd_switch(msg, args)
            case "/delete":
                await self._cmd_delete_mode(msg)
            case "/name":
                await self._cmd_name(msg, args)
            case "/commands":
                if args.strip() == "reload":
                    await self._reload_commands_and_menu(msg)
            case _:
                cmd_name = cmd[1:]  # strip leading "/"

                # Custom Commands (from .gemini/commands/)
                if command := self._cmd_loader.get(cmd_name):
                    logger.info("executing command", cmd=cmd_name, user=msg.user_name)
                    expanded = await self._cmd_loader.expand_prompt(command, args)
                    await self._send_to_agent(msg, expanded)
                    return True

                # Skills (from skill dirs)
                if skill := self._skill_registry.get(cmd_name):
                    logger.info("executing skill", skill=skill.name, user=msg.user_name)
                    prompt = SkillRegistry.build_invocation_prompt(skill, args)
                    await self._send_to_agent(msg, prompt)
                    return True

                await self._reply(msg, self._i18n.t(MsgKey.UNKNOWN_CMD))
                return True
        return True

    async def _process(self, msg: Message, session: Session) -> None:
        """Run agent and stream response back. Drains queued messages after completion."""
        acquired = await session.try_lock()
        if not acquired:
            await self._reply(msg, self._i18n.t(MsgKey.SESSION_BUSY))
            return

        try:
            await self._run_gemini(msg, session)
        finally:
            await session.unlock()
            # Drain queued messages
            q = self._queues.get(msg.session_key)
            if q and not q.empty():
                try:
                    next_msg = q.get_nowait()
                    task = asyncio.create_task(self._process(next_msg, session))
                    task.add_done_callback(
                        lambda t: (
                            logger.error(
                                "Engine: queue drain task failed",
                                exc_info=t.exception(),
                            )
                            if t.exception()
                            and not isinstance(t.exception(), asyncio.CancelledError)
                            else None
                        )
                    )
                except asyncio.QueueEmpty:
                    pass

    async def _run_gemini(self, msg: Message, session: Session) -> None:
        """Run Gemini agent with automatic model-fallback on quota errors."""
        if session.agent_type != "gemini":
            session.agent_type = "gemini"
            self._sessions._save()

        assert msg.reply_ctx is not None
        ctx: ReplyContext = msg.reply_ctx

        for _attempt in range(len(_FALLBACK_MODELS) + 1):
            is_quota = await self._run_agent(msg, session)
            if not is_quota:
                break
            # Quota hit — try next fallback model
            current = self._gemini.model or _FALLBACK_MODELS[0]
            try:
                idx = _FALLBACK_MODELS.index(current)
            except ValueError:
                idx = 0
            if idx + 1 >= len(_FALLBACK_MODELS):
                await self._platform.send(
                    ctx,
                    "⚠️ Tất cả model đều đang quá tải. Vui lòng thử lại sau vài phút."
                )
                break
            next_model = _FALLBACK_MODELS[idx + 1]
            self._gemini.model = next_model
            logger.info("Engine: quota fallback", from_model=current, to_model=next_model)
            await self._platform.send(
                ctx,
                f"⚡ Model `{current}` quá tải, tự động chuyển sang `{next_model}`..."
            )

    async def _run_agent(self, msg: Message, session: Session) -> bool:  # returns True if quota error
        """Send prompt to Gemini or Claude and stream events to Telegram."""
        assert msg.reply_ctx is not None
        ctx: ReplyContext = msg.reply_ctx

        typing_task = await self._platform.start_typing(ctx)

        agent_session = None
        istate = self._interactive.get(msg.session_key)
        agent_type = session.agent_type or "gemini"

        try:
            if agent_type == "claude":
                if self._claude is None:
                    await self._platform.send(ctx, "Claude agent not configured")
                    return
                agent_session = self._claude.start_session(
                    resume_id=session.agent_session_id
                )
                self._active_claude[msg.session_key] = agent_session
            else:
                agent_session = self._gemini.start_session(
                    resume_id=session.agent_session_id
                )
                self._active_gemini[msg.session_key] = agent_session

            preview = StreamPreview(
                config=self._config.stream_preview,
                send_preview=lambda text: self._platform.send_preview_start(ctx, text),
                update_preview=self._platform.update_message,
                delete_preview=self._platform.delete_preview,
            )

            # Track user message in history
            if msg.content:
                session.add_history("user", msg.content, self._sessions.max_history)

            await agent_session.send(
                prompt=msg.content, images=msg.images or [], files=msg.files or []
            )

            full_text = ""
            tool_used = False

            while True:
                try:
                    event = await asyncio.wait_for(
                        agent_session.events.get(), timeout=300
                    )
                except TimeoutError:
                    logger.error("Engine: agent session timed out")
                    break

                match event.type:
                    case EventType.TEXT:
                        if event.session_id:
                            # init/system event: store session_id
                            session.agent_session_id = event.session_id
                        elif event.content:
                            full_text += event.content
                            await preview.append_text(event.content)

                    case EventType.THINKING:
                        # Flush thinking text as a short notification
                        if event.content:
                            max_len = self._config.display.thinking_max_len
                            truncated = (
                                event.content[:max_len] + "…"
                                if len(event.content) > max_len
                                else event.content
                            )
                            thinking_msg = f"<i>{truncated}</i>"
                            await self._platform.send(ctx, thinking_msg)

                    case EventType.TOOL_USE:
                        await preview.freeze()
                        preview.detach()
                        tool_used = True
                        if not (istate and istate.quiet):
                            tool_display = event.tool_input
                            max_len = self._config.display.tool_max_len
                            if len(tool_display) > max_len:
                                tool_display = tool_display[:max_len] + "…"
                            if tool_display:
                                tool_msg = f"🔧 **{event.tool_name}**\n──────────\n{tool_display}"
                            else:
                                tool_msg = f"🔧 **{event.tool_name}**"
                            await self._platform.send(ctx, tool_msg)

                    case EventType.TOOL_RESULT:
                        if (istate and istate.show_tool_output) and not (istate and istate.quiet):
                            if event.content:
                                max_len = self._config.display.tool_max_len
                                output = event.content[:max_len] + "…" if len(event.content) > max_len else event.content
                                await self._platform.send(ctx, f"📤 ```\n{output}\n```")

                    case EventType.PERMISSION_REQUEST:
                        # Claude only: ask user for permission to run tool
                        await preview.freeze()
                        self._pending_permissions[event.request_id] = (
                            _PendingPermission(
                                request_id=event.request_id,
                                tool_name=event.tool_name,
                                tool_input=event.tool_input,
                                ctx=ctx,
                            )
                        )
                        perm_msg = (
                            f"🤔 **{event.tool_name}**\n──────────\n"
                            f"{event.tool_input or 'Permission required'}"
                        )
                        buttons = [
                            ("Allow", f"perm:{event.request_id}:allow"),
                            ("Deny", f"perm:{event.request_id}:deny"),
                        ]
                        await self._platform.send_with_buttons(ctx, perm_msg, buttons)

                    case EventType.ERROR:
                        err_str = str(event.error) if event.error else "❌ Lỗi không xác định."
                        if _is_quota_error(err_str):
                            return True  # signal caller to retry with fallback model
                        await self._platform.send(ctx, err_str)

                    case EventType.RESULT:
                        sent = await preview.finish(full_text)
                        if event.error:
                            err_str = str(event.error)
                            if _is_quota_error(err_str):
                                return True  # quota error in result → retry
                            await self._platform.send(ctx, err_str)
                        elif not sent:
                            if full_text:
                                await self._platform.reply(ctx, full_text)
                            elif not tool_used:
                                await self._platform.send(
                                    ctx, self._i18n.t(MsgKey.EMPTY_RESPONSE)
                                )
                        # Track assistant response in history
                        if full_text:
                            session.add_history(
                                "assistant", full_text, self._sessions.max_history
                            )
                        break

        except Exception as exc:
            logger.exception("Engine: error processing message", error=exc)
            await self._platform.send(ctx, self._i18n.tf(MsgKey.ERROR_PREFIX, str(exc)))
        finally:
            typing_task.cancel()
            self._active_gemini.pop(msg.session_key, None)
            self._active_claude.pop(msg.session_key, None)
            if agent_session:
                await agent_session.close()
        return False  # no quota error

    async def _reply(self, msg: Message, content: str) -> None:
        if msg.reply_ctx is not None:
            ctx: ReplyContext = msg.reply_ctx
            await self._platform.send(ctx, content)

    def _session_key(self, chat_id: int, user_id: str) -> str:
        """Build session key respecting share_session_in_channel config."""
        if self._share_session:
            return f"telegram:{chat_id}:shared"
        return f"telegram:{chat_id}:{user_id}"

    # ── callback routing ──────────────────────────────────────────────────

    async def _handle_cmd_callback(
        self, data: str, user_id: str, chat_id: int, message_id: int
    ) -> None:
        """Re-render command card in-place. data = 'cmd:/list 2'"""
        cmd_with_args = data[4:]  # strip "cmd:"
        parts = cmd_with_args.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""
        session_key = self._session_key(chat_id, user_id)
        ctx = ReplyContext(chat_id=chat_id, message_id=message_id)

        match cmd:
            case "/list":
                card = self._build_list_card(session_key, args)
                await self._platform.edit_card(ctx, message_id, card)
            case "/delete":
                card = self._build_delete_select_card(session_key)
                await self._platform.edit_card(ctx, message_id, card)
            case _:
                logger.debug("Engine: unhandled cmd callback", cmd=cmd)

    async def _handle_act_callback(
        self, data: str, user_id: str, chat_id: int, message_id: int
    ) -> None:
        """Execute action and re-render. data = 'act:cmd:/lang zh'"""
        inner = data[4:]  # strip "act:"
        session_key = self._session_key(chat_id, user_id)
        ctx = ReplyContext(chat_id=chat_id, message_id=message_id)

        if inner.startswith("cmd:"):
            cmd_with_args = inner[4:]
            parts = cmd_with_args.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1].strip() if len(parts) > 1 else ""

            match cmd:
                case "/lang":
                    if args:
                        with contextlib.suppress(ValueError):
                            self._i18n.set_lang(Language(args))
                    card = self._build_lang_card()
                    await self._platform.edit_card(ctx, message_id, card)

                case "/switch":
                    if args:
                        self._sessions.switch_session(session_key, args)
                    card = self._build_list_card(session_key, "")
                    await self._platform.edit_card(ctx, message_id, card)

                case "/delete":
                    match args:
                        case "confirm":
                            istate = self._interactive.get(session_key)
                            if istate and istate.delete_selected:
                                count = self._sessions.delete_sessions(
                                    list(istate.delete_selected)
                                )
                                istate.delete_selected.clear()
                                istate.delete_phase = ""
                                text = self._i18n.tf(MsgKey.SESSION_DELETED, count)
                                card = CardBuilder().markdown(text).build()
                                await self._platform.edit_card(ctx, message_id, card)
                            else:
                                card = self._build_delete_select_card(session_key)
                                await self._platform.edit_card(ctx, message_id, card)
                        case "cancel":
                            istate = self._interactive.get(session_key)
                            if istate:
                                istate.delete_selected.clear()
                                istate.delete_phase = ""
                            text = self._i18n.t(MsgKey.SESSION_DELETE_CANCEL)
                            card = CardBuilder().markdown(text).build()
                            await self._platform.edit_card(ctx, message_id, card)
                        case _:
                            pass

                case "/model":
                    if args:
                        # Set model on the active agent
                        session = self._sessions.get(session_key)
                        agent_type = session.agent_type if session else "gemini"
                        if agent_type == "claude" and self._claude:
                            self._claude.model = args
                        else:
                            self._gemini.model = args
                    card = self._build_model_card(
                        session.agent_type if session else "gemini"
                    )
                    await self._platform.edit_card(ctx, message_id, card)

                case "/delete_one":
                    if args:
                        self._sessions.delete_sessions([args])
                    card = self._build_list_card(session_key, "")
                    await self._platform.edit_card(ctx, message_id, card)

                case _:
                    logger.debug("Engine: unhandled act callback", cmd=cmd)
            return

        logger.debug("Engine: unhandled act callback data", data=data)

    async def _handle_sel_callback(
        self, data: str, user_id: str, chat_id: int, message_id: int
    ) -> None:
        """Toggle selection. data = 'sel:delete:{session_id}'"""
        parts = data.split(":", 3)
        if len(parts) < 3:  # "sel", action, target
            return
        action = parts[1]
        target = parts[2]
        session_key = self._session_key(chat_id, user_id)
        ctx = ReplyContext(chat_id=chat_id, message_id=message_id)

        if action == "delete":
            istate = self._interactive.setdefault(session_key, _InteractiveState())
            istate.delete_phase = "select"
            if target in istate.delete_selected:
                istate.delete_selected.discard(target)
            else:
                istate.delete_selected.add(target)
            card = self._build_delete_select_card(session_key)
            await self._platform.edit_card(ctx, message_id, card)

    async def _handle_perm_callback(
        self, data: str, user_id: str, chat_id: int, message_id: int
    ) -> None:
        """Handle permission response. data = 'perm:{request_id}:{allow|deny}'"""
        parts = data.split(":", 3)
        if len(parts) < 3:
            return
        request_id = parts[1]
        action = parts[2]
        session_key = self._session_key(chat_id, user_id)
        ctx = ReplyContext(chat_id=chat_id, message_id=message_id)

        pending = self._pending_permissions.pop(request_id, None)
        if pending is None:
            logger.debug("Engine: permission not found", request_id=request_id)
            return

        # Find the active claude session for this user
        agent_session = self._active_claude.get(session_key)
        if agent_session is None:
            await self._platform.send(ctx, "No active Claude session")
            return

        allow = action == "allow"
        await agent_session.respond_permission(request_id, allow)

        if allow:
            await self._platform.send(ctx, f"✅ {pending.tool_name} allowed")
        else:
            await self._platform.send(ctx, f"❌ {pending.tool_name} denied")

    # ── card builders ─────────────────────────────────────────────────────

    def _build_lang_card(self) -> Card:
        return (
            CardBuilder()
            .title(self._i18n.tf(MsgKey.LANG_CURRENT, self._i18n.lang.value))
            .actions(
                CardButton("English", "act:cmd:/lang en"),
                CardButton("中文", "act:cmd:/lang zh"),
            )
            .build()
        )

    def _build_model_card(self, agent_type: AgentType = "gemini") -> Card:
        if agent_type == "claude" and self._claude:
            agent = self._claude
            current = agent.model or "(default)"
            models = agent.available_models()
            buttons = [
                CardButton(m.get("desc", ""), f"act:cmd:/model {m['name']}")
                for m in models
            ]
        else:
            agent = self._gemini
            current = agent.model or "(default)"
            models = agent.available_models()
            buttons = [
                CardButton(m.desc or m.name, f"act:cmd:/model {m.name}") for m in models
            ]
        return (
            CardBuilder()
            .title(self._i18n.tf(MsgKey.MODEL_CURRENT, current))
            .actions(*buttons)
            .build()
        )

    def _build_list_card(self, session_key: str, args: str) -> Card:
        page = 1
        with contextlib.suppress(ValueError):
            page = max(1, int(args)) if args else 1

        sessions = self._sessions.list_sessions(session_key)
        total = len(sessions)
        total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        page = min(page, total_pages)
        start = (page - 1) * _PAGE_SIZE
        page_sessions = sessions[start : start + _PAGE_SIZE]
        active_sid = self._sessions.active_session_id(session_key)

        builder = CardBuilder().title(self._i18n.tf(MsgKey.SESSION_LIST_HEADER, total))
        if not sessions:
            builder.note(self._i18n.t(MsgKey.SESSION_LIST_EMPTY))
        else:
            for i, s in enumerate(page_sessions, start=start + 1):
                marker = "▶ " if s.id == active_sid else ""
                buttons: list[CardButton] = [
                    CardButton("🗑️ Delete", f"act:cmd:/delete_one {s.id}")
                ]
                if s.id != active_sid:
                    buttons.insert(0, CardButton("Switch", f"act:cmd:/switch {s.id}"))
                builder.list_item(f"{marker}{i}. {s.summary}", buttons=buttons)
            if total_pages > 1:
                builder.note(self._i18n.tf(MsgKey.PAGE_NAV, page, total_pages))
                nav_btns: list[CardButton] = []
                if page > 1:
                    nav_btns.append(CardButton("◀", f"cmd:/list {page - 1}"))
                if page < total_pages:
                    nav_btns.append(CardButton("▶", f"cmd:/list {page + 1}"))
                if nav_btns:
                    builder.actions(*nav_btns)
        return builder.build()

    def _build_delete_select_card(self, session_key: str) -> Card:
        istate = self._interactive.setdefault(session_key, _InteractiveState())
        sessions = self._sessions.list_sessions(session_key)

        builder = CardBuilder().title(
            self._i18n.tf(MsgKey.SESSION_DELETE_CONFIRM, len(istate.delete_selected))
        )
        if not sessions:
            builder.note(self._i18n.t(MsgKey.SESSION_LIST_EMPTY))
        else:
            for s in sessions:
                marker = "☑ " if s.id in istate.delete_selected else "☐ "
                builder.list_item(
                    f"{marker}{s.summary}", CardButton("Toggle", f"sel:delete:{s.id}")
                )
            builder.actions(
                CardButton("✅ Confirm", "act:cmd:/delete confirm"),
                CardButton("❌ Cancel", "act:cmd:/delete cancel"),
            )
        return builder.build()

    # ── slash command handlers ────────────────────────────────────────────

    async def _cmd_new(self, msg: Message) -> None:
        session = self._sessions.new_session(msg.session_key)
        session.agent_type = "gemini"  # default to gemini for new sessions
        self._sessions._save()
        logger.info(
            "Engine: new session", session_key=msg.session_key, session_id=session.id
        )
        await self._reply(msg, self._i18n.t(MsgKey.SESSION_NEW))

    async def _cmd_help(self, msg: Message) -> None:
        await self._reply(msg, self._i18n.t(MsgKey.HELP))

    async def _cmd_stop(self, msg: Message) -> None:
        gemini = self._active_gemini.get(msg.session_key)
        if gemini:
            await gemini.kill()
            logger.info(
                "Engine: /stop killed active session", session_key=msg.session_key
            )
        await self._reply(msg, self._i18n.t(MsgKey.STOP_OK))

    async def _cmd_model(self, msg: Message, args: str) -> None:
        session = self._sessions.get(msg.session_key)
        agent_type = session.agent_type if session else "gemini"

        if args:
            if agent_type == "claude" and self._claude:
                self._claude.model = args
            else:
                self._gemini.model = args

        if msg.reply_ctx is None:
            return
        card = self._build_model_card(agent_type)
        await self._platform.send_card(msg.reply_ctx, card)

    async def _cmd_mode(self, msg: Message, args: str) -> None:
        session = self._sessions.get(msg.session_key)
        agent_type = session.agent_type if session else "gemini"
        valid_modes = (
            ("default", "auto_edit", "yolo", "plan")
            if agent_type == "gemini"
            else ("default", "acceptEdits", "plan", "bypassPermissions", "dontAsk")
        )
        if args and args in valid_modes:
            if agent_type == "claude" and self._claude:
                self._claude.mode = args
            else:
                self._gemini.mode = args
            await self._reply(msg, self._i18n.tf(MsgKey.MODE_SWITCHED, args))
        elif args:
            modes_str = " | ".join(valid_modes)
            await self._reply(
                msg, self._i18n.tf(MsgKey.MODE_SWITCHED, f"invalid. Use: {modes_str}")
            )
        else:
            if agent_type == "claude" and self._claude:
                current = self._claude.mode
            else:
                current = self._gemini.mode
            await self._reply(msg, self._i18n.tf(MsgKey.MODE_CURRENT, current))

    async def _cmd_agent(self, msg: Message, args: str) -> None:
        """Show or switch the active agent."""
        session = self._sessions.get(msg.session_key)
        current = session.agent_type if session else "gemini"

        if args and args in ("gemini", "claude"):
            if args == "claude" and self._claude is None:
                await self._reply(msg, "Claude agent not configured")
                return
            if session:
                session.agent_type = args  # type: ignore[assignment]
                self._sessions._save()
            await self._reply(msg, f"Agent switched to {args}")
        else:
            await self._reply(msg, f"Current agent: {current} (gemini | claude)")

    async def _cmd_lang(self, msg: Message, args: str) -> None:
        if msg.reply_ctx is None:
            return
        if args:
            try:
                self._i18n.set_lang(Language(args))
                await self._reply(msg, self._i18n.tf(MsgKey.LANG_SWITCHED, args))
            except ValueError:
                await self._reply(msg, f"Unsupported language: {args}")
        else:
            card = self._build_lang_card()
            await self._platform.send_card(msg.reply_ctx, card)

    async def _cmd_quiet(self, msg: Message) -> None:
        istate = self._interactive.setdefault(msg.session_key, _InteractiveState())
        istate.quiet = not istate.quiet
        key = MsgKey.QUIET_ON if istate.quiet else MsgKey.QUIET_OFF
        await self._reply(msg, self._i18n.t(key))

    async def _cmd_toolout(self, msg: Message) -> None:
        istate = self._interactive.setdefault(msg.session_key, _InteractiveState())
        istate.show_tool_output = not istate.show_tool_output
        status = "ON 📤" if istate.show_tool_output else "OFF"
        await self._reply(msg, f"Tool output display: {status}")

    async def _cmd_status(self, msg: Message) -> None:
        if msg.reply_ctx is None:
            return
        session = self._sessions.get(msg.session_key)
        istate = self._interactive.get(msg.session_key)
        agent_type = session.agent_type if session else "gemini"
        if agent_type == "claude" and self._claude:
            model = self._claude.model or "(default)"
            mode = self._claude.mode
        else:
            model = self._gemini.model or "(default)"
            mode = self._gemini.mode
        session_name = session.summary if session else "(none)"
        quiet_icon = "✅" if (istate and istate.quiet) else "❌"
        card = (
            CardBuilder()
            .title("Status")
            .markdown(
                self._i18n.tf(MsgKey.STATUS_INFO, model, mode, session_name, quiet_icon)
                + f"\nAgent: {agent_type}"
            )
            .build()
        )
        await self._platform.send_card(msg.reply_ctx, card)

    async def _cmd_list(self, msg: Message, args: str) -> None:
        if msg.reply_ctx is None:
            return
        card = self._build_list_card(msg.session_key, args)
        await self._platform.send_card(msg.reply_ctx, card)

    async def _cmd_current(self, msg: Message) -> None:
        session = self._sessions.get(msg.session_key)
        if session:
            text = self._i18n.tf(MsgKey.SESSION_CURRENT, session.summary)
        else:
            text = self._i18n.t(MsgKey.SESSION_LIST_EMPTY)
        await self._reply(msg, text)

    async def _cmd_history(self, msg: Message) -> None:
        session = self._sessions.get(msg.session_key)
        if not session or not session.history:
            await self._reply(msg, self._i18n.t(MsgKey.SESSION_HISTORY_EMPTY))
            return
        header = self._i18n.t(MsgKey.SESSION_HISTORY_HEADER)
        entries = session.history[-_HISTORY_DISPLAY:]
        lines = [
            f"[{e.role}] {e.content[:80]}{'…' if len(e.content) > 80 else ''}"
            for e in entries
        ]
        await self._reply(msg, header + "\n" + "\n".join(lines))

    async def _cmd_switch(self, msg: Message, args: str) -> None:
        if not args:
            await self._reply(msg, self._i18n.tf(MsgKey.SESSION_NOT_FOUND, "(none)"))
            return
        session = self._sessions.switch_session(msg.session_key, args)
        if session:
            await self._reply(
                msg, self._i18n.tf(MsgKey.SESSION_SWITCHED, session.summary)
            )
        else:
            await self._reply(msg, self._i18n.tf(MsgKey.SESSION_NOT_FOUND, args))

    async def _cmd_delete_mode(self, msg: Message) -> None:
        if msg.reply_ctx is None:
            return
        istate = self._interactive.setdefault(msg.session_key, _InteractiveState())
        istate.delete_selected.clear()
        istate.delete_phase = "select"
        card = self._build_delete_select_card(msg.session_key)
        await self._platform.send_card(msg.reply_ctx, card)

    async def _cmd_name(self, msg: Message, args: str) -> None:
        session = self._sessions.get(msg.session_key)
        if not session:
            await self._reply(msg, self._i18n.t(MsgKey.SESSION_LIST_EMPTY))
            return
        new_name = args.strip()
        self._sessions.set_session_name(session.id, new_name)
        await self._reply(msg, self._i18n.tf(MsgKey.SESSION_NAMED, new_name))

    # ── Skills + Commands helpers ─────────────────────────────────────────

    async def _send_to_agent(self, msg: Message, content: str) -> None:
        """Forward expanded skill/command prompt to the active agent."""
        new_msg = Message(
            session_key=msg.session_key,
            platform=msg.platform,
            user_id=msg.user_id,
            user_name=msg.user_name,
            content=content,
            reply_ctx=msg.reply_ctx,
        )
        session = self._sessions.get_or_create(msg.session_key)
        await self._process(new_msg, session)

    async def _reload_commands_and_menu(self, msg: Message) -> None:
        """Reload commands and skills, then refresh Telegram command menu."""
        cmd_count = self._cmd_loader.reload()
        self._skill_registry.invalidate()
        skill_count = self._skill_registry.load()
        logger.info(
            f"reloaded commands and skills: {cmd_count} commands, {skill_count} skills"
        )
        await self._reply(msg, f"Reloaded: {cmd_count} commands, {skill_count} skills")
        await self._refresh_commands_menu()

    async def _refresh_commands_menu(self) -> None:
        """Build the full command list and push it to Telegram."""
        commands: list[tuple[str, str]] = [
            ("new", "Start new session"),
            ("list", "List all sessions"),
            ("switch", "Switch active session"),
            ("current", "Show current session"),
            ("history", "Show conversation history"),
            ("name", "Rename session"),
            ("delete", "Delete sessions"),
            ("status", "Show status"),
            ("model", "Show/switch model"),
            ("mode", "Show/switch mode"),
            ("agent", "Switch agent (gemini/claude)"),
            ("lang", "Switch language"),
            ("quiet", "Toggle quiet mode"),
            ("toolout", "Toggle tool output display"),
            ("stop", "Stop agent"),
            ("help", "Show help"),
            ("commands", "Reload commands: /commands reload"),
        ]
        # Commands before Skills; prefix distinguishes them in description
        commands.extend(
            (_to_tg_command(cmd.name), f"[CMD] {cmd.description}")
            for cmd in self._cmd_loader.list_all()
        )
        commands.extend(
            (_to_tg_command(skill.name), f"[SKILL] {skill.description}")
            for skill in self._skill_registry.list_all()
        )
        await self._platform.set_commands_menu(commands)
