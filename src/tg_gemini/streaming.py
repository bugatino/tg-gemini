"""Throttled stream preview: accumulates text and batches Telegram message edits."""

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from tg_gemini.config import StreamPreviewConfig

__all__ = ["StreamPreview"]


class StreamPreview:
    """Accumulates streaming text and throttles preview updates to a single Telegram message.

    Lifecycle:
        1. append_text() — called for each EventText chunk; schedules/sends preview updates
        2. freeze() — called on tool_use; stops updates but leaves preview message visible
        3. finish(final_text) — called on EventResult; sends the final text update
        4. detach() — clear preview handle (keep message but stop tracking)
    """

    def __init__(
        self,
        config: StreamPreviewConfig,
        send_preview: Callable[[str], Awaitable[Any]],
        update_preview: Callable[[Any, str], Awaitable[None]],
        delete_preview: Callable[[Any], Awaitable[None]],
    ) -> None:
        self._cfg = config
        self._send_preview = send_preview
        self._update_preview = update_preview
        self._delete_preview = delete_preview

        self._full_text = ""
        self._last_sent_text = ""
        self._last_sent_at: float = 0.0
        self._last_sent_via_update = False
        self._preview_handle: Any = None
        self._degraded = False
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    @property
    def full_text(self) -> str:
        return self._full_text

    async def append_text(self, text: str) -> None:
        """Accumulate text and trigger a throttled preview flush."""
        if not self._cfg.enabled:
            return
        async with self._lock:
            self._full_text += text
            # When degraded, accumulate text but don't send previews;
            # finish() will use reply() as fallback
            if self._degraded:
                return
            display = self._trimmed_text(self._full_text)
            delta = len(display) - len(self._last_sent_text)
            elapsed_ms = (time.monotonic() - self._last_sent_at) * 1000
            interval_ms = self._cfg.interval_ms

            if self._last_sent_at > 0:
                if delta < self._cfg.min_delta_chars:
                    self._schedule_flush(interval_ms)
                    return
                if elapsed_ms < interval_ms:
                    self._schedule_flush(interval_ms - elapsed_ms)
                    return

            self._cancel_flush()
            await self._flush_locked(display)

    def _trimmed_text(self, text: str) -> str:
        max_chars = self._cfg.max_chars
        runes = list(text)
        if max_chars > 0 and len(runes) > max_chars:
            return "".join(runes[:max_chars]) + "…"
        return text

    def _schedule_flush(self, delay_ms: float) -> None:
        if self._flush_task is not None:
            return  # already scheduled
        delay_secs = max(delay_ms / 1000, 0.01)

        async def _do_flush() -> None:
            await asyncio.sleep(delay_secs)
            async with self._lock:
                self._flush_task = None
                if self._degraded:
                    return
                display = self._trimmed_text(self._full_text)
                await self._flush_locked(display)

        self._flush_task = asyncio.create_task(_do_flush())

    def _cancel_flush(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None

    async def _flush_locked(self, display: str) -> None:
        """Send or update preview message. Must be called under self._lock."""
        if display == self._last_sent_text or not display:
            return

        if self._preview_handle is None:
            try:
                handle = await self._send_preview(display)
                self._preview_handle = handle
                self._last_sent_text = display
                self._last_sent_via_update = False
                self._last_sent_at = time.monotonic()
                logger.debug("StreamPreview: sent initial preview")
            except Exception as exc:
                logger.debug("StreamPreview: initial send failed, degrading", error=exc)
                self._degraded = True
        else:
            try:
                await self._update_preview(self._preview_handle, display)
                self._last_sent_text = display
                self._last_sent_via_update = True
                self._last_sent_at = time.monotonic()
                logger.debug("StreamPreview: updated preview")
            except Exception as exc:
                logger.debug("StreamPreview: update failed, degrading", error=exc)
                self._degraded = True

    async def freeze(self) -> None:
        """Stop preview updates (called before a tool_use notification)."""
        async with self._lock:
            self._cancel_flush()
            if self._preview_handle is not None and not self._degraded:
                text = self._trimmed_text(self._full_text)
                if text:
                    with contextlib.suppress(Exception):
                        await self._update_preview(self._preview_handle, text)
            self._degraded = True

    async def finish(self, final_text: str) -> bool:
        """Send the final update. Returns True if preview was used (caller can skip re-sending)."""
        async with self._lock:
            self._cancel_flush()

            if self._preview_handle is None or self._degraded:
                if self._preview_handle is not None and self._degraded:
                    with contextlib.suppress(Exception):
                        await self._delete_preview(self._preview_handle)
                return False

            if not final_text:
                return False

            if final_text == self._last_sent_text and self._last_sent_via_update:
                logger.debug("StreamPreview: final text unchanged, skipping update")
                return True

            try:
                await self._update_preview(self._preview_handle, final_text)
            except Exception as exc:
                logger.debug("StreamPreview: final update failed", error=exc)
                with contextlib.suppress(Exception):
                    await self._delete_preview(self._preview_handle)
                return False
            else:
                logger.debug("StreamPreview: final update sent")
                return True

    def detach(self) -> None:
        """Clear the preview handle (keep message visible but stop tracking)."""
        self._preview_handle = None

    async def delete(self) -> None:
        """Delete the preview message and stop tracking (called before tool notifications).

        Unlike freeze(), this removes the pre-tool text from chat entirely,
        preventing a duplicate greeting when the final response is sent.
        """
        async with self._lock:
            self._cancel_flush()
            if self._preview_handle is not None:
                with contextlib.suppress(Exception):
                    await self._delete_preview(self._preview_handle)
            self._degraded = True
            self._preview_handle = None
