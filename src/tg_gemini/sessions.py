"""Async JSON persistence for per-user session metadata."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_SESSIONS_PATH = Path.home() / ".config" / "tg-gemini" / "sessions.json"


@dataclass(frozen=True)
class PersistedSession:
    """Subset of UserSession fields that survive a restart."""

    session_id: str | None = None
    model: str | None = None
    custom_names: dict[str, str] = field(default_factory=dict)


class SessionStore:
    """Async-safe JSON store for session metadata.

    All I/O runs on a thread pool via asyncio.to_thread so it never blocks
    the event loop. Writes use atomic rename (temp + replace) to survive crashes.
    """

    def __init__(
        self, *, _path: Path | None = None, _lock: asyncio.Lock | None = None
    ) -> None:
        self._path: Path = _path if _path is not None else DEFAULT_SESSIONS_PATH
        self._lock: asyncio.Lock = _lock if _lock is not None else asyncio.Lock()

    # ── Public API ───────────────────────────────────────────────────────────

    async def load(self) -> dict[int, PersistedSession]:
        """Load all persisted sessions from disk.

        Returns:
            Mapping of user_id (int) → PersistedSession.
            Empty dict if the file is missing, corrupt, or has wrong type.
        """
        if not self._path.exists():
            logger.debug("sessions_file_missing", path=str(self._path))
            return {}

        try:
            raw: dict = await asyncio.to_thread(self._read_json, self._path)
        except Exception:
            logger.warning("sessions_load_failed", path=str(self._path))
            return {}

        if not isinstance(raw, dict):
            logger.warning("sessions_invalid_root_type", path=str(self._path))
            return {}

        result: dict[int, PersistedSession] = {}
        for uid_str, data in raw.items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                logger.debug("sessions_skip_invalid_user_id", uid=uid_str)
                continue
            result[uid] = self._deserialize(data, uid)

        logger.info("sessions_loaded", count=len(result))
        return result

    async def save(self, user_id: int, session: PersistedSession) -> None:
        """Persist a single user's session data to disk.

        Performs a read-modify-write of the full file under _lock
        so concurrent saves from multiple users don't corrupt each other.
        """
        async with self._lock:
            try:
                raw: dict = await asyncio.to_thread(self._read_json, self._path)
            except Exception:
                raw = {}

            if not isinstance(raw, dict):
                raw = {}

            raw[str(user_id)] = self._serialize(session)
            await asyncio.to_thread(self._write_json_atomic, self._path, raw)
            logger.debug("session_saved", user_id=user_id)

    async def save_all(self, sessions: dict[int, PersistedSession]) -> None:
        """Atomically overwrite the entire sessions file (used on shutdown)."""
        async with self._lock:
            raw = {str(uid): self._serialize(s) for uid, s in sessions.items()}
            await asyncio.to_thread(self._write_json_atomic, self._path, raw)
            logger.debug("sessions_saved_all", count=len(raw))

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _read_json(path: Path) -> dict:
        """Read and parse a JSON file. Runs on a thread pool."""
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _write_json_atomic(path: Path, data: dict) -> None:
        """Write JSON atomically via temp file + rename.

        Atomic on POSIX; best-effort on Windows (same-filesystem rename is atomic
        on Python 3.11+ for both POSIX and Windows).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _serialize(s: PersistedSession) -> dict:
        return {
            "session_id": s.session_id,
            "model": s.model,
            "custom_names": s.custom_names,
        }

    @staticmethod
    def _deserialize(data: dict, user_id: int) -> PersistedSession:
        """Deserialize a persisted session, handling missing/extra keys gracefully."""
        if not isinstance(data, dict):
            logger.debug("sessions_invalid_user_data", user_id=user_id)
            return PersistedSession()
        custom_names_val = data.get("custom_names")
        return PersistedSession(
            session_id=data.get("session_id") if "session_id" in data else None,
            model=data.get("model") if "model" in data else None,
            custom_names=custom_names_val if isinstance(custom_names_val, dict) else {},
        )
