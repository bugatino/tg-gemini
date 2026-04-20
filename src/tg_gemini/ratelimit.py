"""Sliding-window rate limiter for per-user request throttling."""

import asyncio
import contextlib
import time

__all__ = ["RateLimiter"]


class RateLimiter:
    """Sliding window rate limiter keyed by arbitrary strings.

    Set ``max_messages=0`` (default) to disable rate limiting entirely.
    """

    def __init__(
        self,
        max_messages: int = 0,
        window_secs: float = 60.0,
        cleanup_interval_secs: float = 300.0,
    ) -> None:
        self._max = max_messages
        self._window = window_secs
        self._cleanup_interval = cleanup_interval_secs
        self._buckets: dict[str, list[float]] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    def allow(self, key: str) -> bool:
        """Return True if the request is within the limit (or limiting is disabled)."""
        if self._max == 0:
            return True
        now = time.monotonic()
        cutoff = now - self._window
        timestamps = [t for t in self._buckets.get(key, []) if t > cutoff]
        if len(timestamps) >= self._max:
            self._buckets[key] = timestamps
            return False
        timestamps.append(now)
        self._buckets[key] = timestamps
        return True

    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Cancel the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval)
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove buckets whose timestamps have all expired."""
        now = time.monotonic()
        cutoff = now - self._window
        stale = [k for k, ts in self._buckets.items() if all(t <= cutoff for t in ts)]
        for k in stale:
            del self._buckets[k]
