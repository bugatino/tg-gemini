"""Telegram bot platform using python-telegram-bot v21+ async API."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from telegram import Bot, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from tg_gemini.card import Card
from tg_gemini.markdown import markdown_to_html, split_message
from tg_gemini.models import (
    FileAttachment,
    ImageAttachment,
    Message as CoreMessage,
    PreviewHandle,
    ReplyContext,
)

__all__ = ["TelegramPlatform"]

MessageHandlerType = Callable[[CoreMessage], Awaitable[None]]

_TG_MAX_LEN = 4096
_OLD_MESSAGE_SECONDS = 30


def _is_allowed(allow_from: str, user_id: str) -> bool:
    if allow_from == "*":
        return True
    allowed = [x.strip() for x in allow_from.split(",")]
    return user_id in allowed


class TelegramPlatform:
    """Manages Telegram polling, message routing, and sending."""

    def __init__(
        self,
        token: str,
        allow_from: str = "*",
        group_reply_all: bool = False,
        share_session_in_channel: bool = False,
    ) -> None:
        self._token = token
        self._allow_from = allow_from
        self._group_reply_all = group_reply_all
        self._share_session_in_channel = share_session_in_channel
        self._bot_id: str = ""
        self._bot_username: str = ""
        self._app: Application[Any, Any, Any, Any, Any, Any] | None = None
        # exact-match callbacks: data → handler(data, user_id, chat_id, message_id)
        self._callback_handlers: dict[
            str, Callable[[str, str, int, int], Awaitable[None]]
        ] = {}
        # prefix-match callbacks: prefix → handler(data, user_id, chat_id, message_id)
        self._prefix_handlers: dict[
            str, Callable[[str, str, int, int], Awaitable[None]]
        ] = {}
        self._message_handler: MessageHandlerType | None = None

    async def _handle_update(self, update: Update, context: Any) -> None:  # noqa: ARG002
        """Process a Telegram Update containing a message."""
        msg = update.message
        if msg is None:
            return

        # Ignore old messages (pre-startup)
        if time.time() - msg.date.timestamp() > _OLD_MESSAGE_SECONDS:
            logger.debug("TelegramPlatform: ignoring old message")
            return

        from_ = msg.from_user
        if from_ is None:
            return

        user_id = str(from_.id)
        if not _is_allowed(self._allow_from, user_id):
            logger.debug("TelegramPlatform: unauthorized user", user_id=user_id)
            return

        user_name = from_.username or f"{from_.first_name} {from_.last_name}".strip()
        chat_id = msg.chat_id
        is_group = msg.chat.type in ("group", "supergroup")

        # Group chat filtering: skip non-directed messages unless group_reply_all
        if is_group and not self._group_reply_all:
            text = msg.text or msg.caption or ""
            if not text.startswith("/"):
                mentioned = self._bot_username and f"@{self._bot_username}" in text
                reply_to = msg.reply_to_message
                replied_to_bot = (
                    reply_to is not None
                    and reply_to.from_user is not None
                    and self._bot_id
                    and str(reply_to.from_user.id) == self._bot_id
                )
                if not mentioned and not replied_to_bot:
                    logger.debug("TelegramPlatform: group message not directed at bot")
                    return

        # Session key: shared across group or per-user
        if is_group and self._share_session_in_channel:
            session_key = f"telegram:{chat_id}:shared"
        else:
            session_key = f"telegram:{chat_id}:{from_.id}"

        reply_ctx = ReplyContext(chat_id=chat_id, message_id=msg.message_id)

        if self._app is None:
            return
        bot: Bot = self._app.bot

        # Handle photo messages
        if msg.photo:
            best = msg.photo[-1]
            try:
                photo_file = await bot.get_file(best.file_id)
                data = await photo_file.download_as_bytearray()
            except Exception as exc:
                logger.error("TelegramPlatform: download photo failed", error=exc)
                return
            caption = (msg.caption or "").strip()
            core_msg = CoreMessage(
                session_key=session_key,
                platform="telegram",
                user_id=user_id,
                user_name=user_name,
                content=caption,
                message_id=str(msg.message_id),
                images=[ImageAttachment(mime_type="image/jpeg", data=bytes(data))],
                reply_ctx=reply_ctx,
            )
            if self._message_handler:
                await self._message_handler(core_msg)
            return

        # Handle document messages
        if msg.document:
            doc = msg.document
            try:
                doc_file = await bot.get_file(doc.file_id)
                data = await doc_file.download_as_bytearray()
            except Exception as exc:
                logger.error("TelegramPlatform: download document failed", error=exc)
                return
            caption = (msg.caption or "").strip()
            core_msg = CoreMessage(
                session_key=session_key,
                platform="telegram",
                user_id=user_id,
                user_name=user_name,
                content=caption,
                message_id=str(msg.message_id),
                files=[
                    FileAttachment(
                        mime_type=doc.mime_type or "application/octet-stream",
                        data=bytes(data),
                        file_name=doc.file_name or "",
                    )
                ],
                reply_ctx=reply_ctx,
            )
            if self._message_handler:
                await self._message_handler(core_msg)
            return

        # Text messages
        text = (msg.text or "").strip()
        if not text:
            return

        core_msg = CoreMessage(
            session_key=session_key,
            platform="telegram",
            user_id=user_id,
            user_name=user_name,
            content=text,
            message_id=str(msg.message_id),
            reply_ctx=reply_ctx,
        )
        if self._message_handler:
            await self._message_handler(core_msg)

    async def _handle_callback(self, update: Update, context: Any) -> None:  # noqa: ARG002
        """Process an inline keyboard button press."""
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        data = query.data or ""
        from_ = query.from_user
        if from_ is None:
            return
        user_id = str(from_.id)
        msg = query.message
        chat_id = int(msg.chat.id) if msg is not None else 0
        message_id = int(msg.message_id) if msg is not None else 0
        # Exact match
        cb = self._callback_handlers.get(data)
        if cb:
            await cb(data, user_id, chat_id, message_id)
            return
        # Prefix match
        for prefix, handler in self._prefix_handlers.items():
            if data.startswith(prefix):
                await handler(data, user_id, chat_id, message_id)
                return

    async def start(
        self,
        handler: MessageHandlerType,
        on_started: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Start long-polling. Blocks until stop() is called."""
        self._message_handler = handler
        self._app = Application.builder().token(self._token).build()
        app = self._app

        # Drain old updates on startup (offset=-1 marks all as read)
        try:
            bot: Bot = app.bot
            await bot.get_updates(offset=-1, timeout=0)
        except Exception as exc:
            logger.warning("TelegramPlatform: failed to drain old updates", error=exc)

        # Cache bot identity for group-chat mention filtering
        try:
            me = await app.bot.get_me()
            self._bot_id = str(me.id)
            self._bot_username = me.username or ""
            logger.info("bot started", username=self._bot_username)
        except Exception as exc:
            logger.warning("TelegramPlatform: failed to fetch bot info", error=exc)

        app.add_handler(
            MessageHandler(filters.ALL & ~filters.COMMAND, self._handle_update)
        )
        app.add_handler(
            CommandHandler(
                [
                    "start",
                    "stop",
                    "new",
                    "resume",
                    "model",
                    "mode",
                    "help",
                    "lang",
                    "quiet",
                    "status",
                    "list",
                    "current",
                    "history",
                    "switch",
                    "delete",
                    "name",
                    "commands",
                ],
                self._handle_update,
            )
        )
        app.add_handler(CallbackQueryHandler(self._handle_callback))

        async with app:
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True)  # type: ignore[union-attr]
            if on_started:
                await on_started()
            try:
                while self._app is not None:
                    await asyncio.sleep(0.5)
            finally:
                await app.updater.stop()
                await app.stop()

    async def stop(self) -> None:
        """Signal the platform to stop polling."""
        self._app = None

    async def set_commands_menu(self, commands: list[tuple[str, str]]) -> None:
        """Register the bot command list shown in the Telegram UI."""
        if self._app is None:
            return
        bot_commands = [BotCommand(name, desc[:100]) for name, desc in commands]
        try:
            await self._app.bot.set_my_commands(bot_commands)
            logger.info(f"commands menu set: {len(bot_commands)} entries")
            return
        except Exception as exc:
            logger.warning(
                f"failed to set commands menu, retrying...{exc:}, type{type(exc).__name__}",
                error=exc,
                error_type=type(exc).__name__,
            )
            for cmd in bot_commands:
                logger.debug(f"  command: {cmd.command} - {cmd.description}")

        # Retry once after a short delay (handles timing/permission flap)
        await asyncio.sleep(2)
        try:
            await self._app.bot.set_my_commands(bot_commands)
            logger.info(f"commands menu set (retry): {len(bot_commands)} entries")
        except Exception as exc2:
            logger.error(
                "failed to set commands menu after retry",
                error=exc2,
                error_type=type(exc2).__name__,
            )

    async def reply(self, ctx: ReplyContext, content: str) -> None:
        """Send a reply, converting markdown to HTML. Splits if >4096 chars."""
        await self._send_html(
            ctx.chat_id, markdown_to_html(content), reply_to=ctx.message_id
        )

    async def send(self, ctx: ReplyContext, content: str) -> None:
        """Send a message (no reply threading)."""
        await self._send_html(ctx.chat_id, markdown_to_html(content))

    async def send_image(self, ctx: ReplyContext, img: ImageAttachment) -> None:
        if self._app is None:
            return
        try:
            await self._app.bot.send_photo(chat_id=ctx.chat_id, photo=img.data)
        except Exception as exc:
            logger.error("TelegramPlatform: send_image failed", error=exc)

    async def send_file(self, ctx: ReplyContext, file: FileAttachment) -> None:
        if self._app is None:
            return
        try:
            await self._app.bot.send_document(
                chat_id=ctx.chat_id,
                document=file.data,
                filename=file.file_name or "file",
            )
        except Exception as exc:
            logger.error("TelegramPlatform: send_file failed", error=exc)

    async def send_preview_start(
        self, ctx: ReplyContext, content: str
    ) -> PreviewHandle:
        """Send initial streaming preview message. Returns handle for updates."""
        if self._app is None:
            raise RuntimeError("Platform not started")
        html = markdown_to_html(content)
        msg = await self._app.bot.send_message(
            chat_id=ctx.chat_id, text=html or "…", parse_mode=ParseMode.HTML
        )
        return PreviewHandle(chat_id=ctx.chat_id, message_id=msg.message_id)

    async def update_message(self, handle: PreviewHandle, content: str) -> None:
        """Edit an existing preview message in place."""
        if self._app is None:
            return
        html = markdown_to_html(content)
        try:
            await self._app.bot.edit_message_text(
                chat_id=handle.chat_id,
                message_id=handle.message_id,
                text=html or "…",
                parse_mode=ParseMode.HTML,
            )
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise
        except RetryAfter as exc:
            logger.warning(
                "TelegramPlatform: rate limited, retry after",
                retry_after=exc.retry_after,
            )
            # Don't raise - the preview will be cleaned up by finish() and reply() will be used

    async def delete_preview(self, handle: PreviewHandle) -> None:
        """Delete a preview message."""
        if self._app is None:
            return
        try:
            await self._app.bot.delete_message(
                chat_id=handle.chat_id, message_id=handle.message_id
            )
        except Exception as exc:
            logger.debug("TelegramPlatform: delete_preview failed", error=exc)

    async def start_typing(self, ctx: ReplyContext) -> asyncio.Task[None]:
        """Start sending typing indicators in the background. Returns a task to cancel."""

        async def _typing_loop() -> None:
            if self._app is None:
                return
            try:
                while True:
                    await self._app.bot.send_chat_action(
                        chat_id=ctx.chat_id, action=ChatAction.TYPING
                    )
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("TelegramPlatform: typing loop error", error=exc)

        return asyncio.create_task(_typing_loop())

    async def send_with_buttons(
        self, ctx: ReplyContext, content: str, buttons: list[tuple[str, str]]
    ) -> None:
        """Send a message with inline keyboard buttons [(label, callback_data), ...]."""
        if self._app is None:
            return
        keyboard = [
            [InlineKeyboardButton(label, callback_data=data)] for label, data in buttons
        ]
        markup = InlineKeyboardMarkup(keyboard)
        html = markdown_to_html(content)
        try:
            await self._app.bot.send_message(
                chat_id=ctx.chat_id,
                text=html or content,
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
            )
        except Exception as exc:
            logger.error("TelegramPlatform: send_with_buttons failed", error=exc)

    def register_callback_prefix(
        self, prefix: str, handler: Callable[[str, str, int, int], Awaitable[None]]
    ) -> None:
        """Register a prefix-based callback handler.

        Handler signature: (data, user_id, chat_id, message_id) → Awaitable[None].
        """
        self._prefix_handlers[prefix] = handler

    async def send_card(self, ctx: ReplyContext, card: Card) -> int:
        """Send a Card as HTML + optional InlineKeyboardMarkup. Returns message_id."""
        if self._app is None:
            return 0
        html = card.render_text() or "…"
        if card.has_buttons():
            rows = card.collect_buttons()
            keyboard = [
                [
                    InlineKeyboardButton(b.text, callback_data=b.callback_data)
                    for b in row
                ]
                for row in rows
            ]
            markup = InlineKeyboardMarkup(keyboard)
            try:
                sent = await self._app.bot.send_message(
                    chat_id=ctx.chat_id,
                    text=html,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
                return sent.message_id
            except Exception as exc:
                logger.error("TelegramPlatform: send_card failed", error=exc)
                return 0
        await self._send_html(ctx.chat_id, html)
        return 0

    async def edit_card(self, ctx: ReplyContext, message_id: int, card: Card) -> None:
        """Edit an existing message in-place with new Card content."""
        if self._app is None:
            return
        html = card.render_text() or "…"
        if card.has_buttons():
            rows = card.collect_buttons()
            keyboard = [
                [
                    InlineKeyboardButton(b.text, callback_data=b.callback_data)
                    for b in row
                ]
                for row in rows
            ]
            markup = InlineKeyboardMarkup(keyboard)
        else:
            markup = InlineKeyboardMarkup([])
        try:
            await self._app.bot.edit_message_text(
                chat_id=ctx.chat_id,
                message_id=message_id,
                text=html,
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
            )
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                logger.error("TelegramPlatform: edit_card failed", error=exc)
        except Exception as exc:
            logger.error("TelegramPlatform: edit_card failed", error=exc)

    async def _send_html(
        self, chat_id: int, html: str, reply_to: int | None = None
    ) -> None:
        if self._app is None:
            return
        chunks = split_message(html, _TG_MAX_LEN)
        for i, chunk in enumerate(chunks):
            try:
                kwargs: dict[str, Any] = {
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": ParseMode.HTML,
                }
                if i == 0 and reply_to:
                    kwargs["reply_to_message_id"] = reply_to
                await self._app.bot.send_message(**kwargs)
            except BadRequest as exc:
                logger.warning(
                    "TelegramPlatform: HTML parse failed, sending plain", error=exc
                )
                try:
                    plain_kwargs: dict[str, Any] = {"chat_id": chat_id, "text": chunk}
                    if i == 0 and reply_to:
                        plain_kwargs["reply_to_message_id"] = reply_to
                    await self._app.bot.send_message(**plain_kwargs)
                except Exception as exc2:
                    logger.error("TelegramPlatform: send plain also failed", error=exc2)
