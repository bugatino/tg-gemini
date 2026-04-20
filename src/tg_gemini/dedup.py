"""Message deduplication — prevents double-processing repeated Telegram updates."""

import time

__all__ = ["MessageDedup"]


class MessageDedup:
    """TTL-based deduplication keyed by message ID.

    Empty IDs are never considered duplicates (pass-through).
    """

    def __init__(self, ttl_secs: float = 60.0) -> None:
        self._ttl = ttl_secs
        self._seen: dict[str, float] = {}

    def is_duplicate(self, msg_id: str) -> bool:
        """Return True if *msg_id* was already seen within the TTL window.

        First occurrence → False (not duplicate, records the ID).
        Repeated occurrence within TTL → True (duplicate).
        Empty *msg_id* → always False (not tracked).
        """
        if not msg_id:
            return False
        now = time.monotonic()
        if msg_id in self._seen and now - self._seen[msg_id] < self._ttl:
            return True
        self._seen[msg_id] = now
        self._clean_expired()
        return False

    def _clean_expired(self) -> None:
        """Remove entries older than TTL."""
        now = time.monotonic()
        stale = [k for k, t in self._seen.items() if now - t >= self._ttl]
        for k in stale:
            del self._seen[k]
