"""Session management for Telegram bot users.

v2: N sessions per user, one "active" at a time, with conversation history.
Migration from v1 single-session JSON format is handled automatically on load.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tg_gemini.config import AgentType

__all__ = ["HistoryEntry", "Session", "SessionManager"]


@dataclass
class HistoryEntry:
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Session:
    """A user session with locking capability and conversation history."""

    id: str
    user_key: str = ""
    agent_session_id: str = ""
    agent_type: AgentType = "gemini"  # which agent this session uses
    name: str = ""
    history: list[HistoryEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _busy: bool = field(default=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def try_lock(self) -> bool:
        """Try to acquire the session lock. Returns False if already busy."""
        async with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    async def unlock(self) -> None:
        """Release the session lock and update timestamp."""
        async with self._lock:
            self._busy = False
            self.updated_at = datetime.now(UTC)

    @property
    def busy(self) -> bool:
        return self._busy

    def add_history(self, role: str, content: str, max_entries: int = 50) -> None:
        """Append a history entry, trimming oldest when over limit."""
        self.history.append(HistoryEntry(role=role, content=content))
        if max_entries > 0 and len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    @property
    def summary(self) -> str:
        """Human-readable label: name → first user message preview → truncated ID."""
        if self.name:
            return self.name
        for entry in self.history:
            if entry.role == "user" and entry.content:
                preview = entry.content[:30]
                return preview + "…" if len(entry.content) > 30 else preview
        return self.id[:8]


class SessionManager:
    """Manages multiple sessions per user with optional JSON persistence.

    v1 API (backward-compatible):
        get_or_create(user_key) → Session
        new_session(user_key, name="") → Session
        get(user_key) → Session | None

    v2 additions:
        list_sessions / switch_session / delete_session / delete_sessions
        set_session_name / find_session / active_session_id / session_count
    """

    def __init__(self, store_path: Path | None = None, max_history: int = 50) -> None:
        self._sessions: dict[str, Session] = {}  # session_id → Session
        self._active: dict[str, str] = {}  # user_key → active session_id
        self._user_sessions: dict[str, list[str]] = {}  # user_key → [session_id, ...]
        self._counter: int = 0
        self._store_path = store_path
        self._max_history = max_history
        if store_path and store_path.exists():
            self._load()

    @property
    def max_history(self) -> int:
        return self._max_history

    # ── v1 API ────────────────────────────────────────────────────────────

    def get_or_create(self, user_key: str) -> Session:
        """Get the active session, creating one if none exists."""
        sid = self._active.get(user_key)
        if sid and sid in self._sessions:
            return self._sessions[sid]
        return self._create_session(user_key)

    def new_session(self, user_key: str, name: str = "") -> Session:
        """Create a fresh session for user_key and make it active."""
        session = self._create_session(user_key, name=name)
        self._save()
        return session

    def get(self, user_key: str) -> Session | None:
        """Get the active session for a user, or None."""
        sid = self._active.get(user_key)
        return self._sessions.get(sid) if sid else None

    # ── v2 API ────────────────────────────────────────────────────────────

    def list_sessions(self, user_key: str) -> list[Session]:
        """Return sessions for user sorted by updated_at descending."""
        sids = self._user_sessions.get(user_key, [])
        sessions = [self._sessions[sid] for sid in sids if sid in self._sessions]
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def switch_session(self, user_key: str, target: str) -> Session | None:
        """Switch active session by 1-based index, ID prefix, or name substring."""
        sessions = self.list_sessions(user_key)
        if not sessions:
            return None
        # 1-based index
        try:
            idx = int(target) - 1
            if 0 <= idx < len(sessions):
                self._active[user_key] = sessions[idx].id
                self._save()
                return sessions[idx]
        except ValueError:
            pass
        # ID prefix
        for s in sessions:
            if s.id.startswith(target):
                self._active[user_key] = s.id
                self._save()
                return s
        # Name substring
        tl = target.lower()
        for s in sessions:
            if tl in s.name.lower():
                self._active[user_key] = s.id
                self._save()
                return s
        return None

    def active_session_id(self, user_key: str) -> str:
        return self._active.get(user_key, "")

    def set_session_name(self, session_id: str, name: str) -> bool:
        """Rename a session. Returns False if not found."""
        if session_id not in self._sessions:
            return False
        self._sessions[session_id].name = name
        self._save()
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. If active, promote next most-recent. Returns False if not found."""
        if session_id not in self._sessions:
            return False
        session = self._sessions.pop(session_id)
        user_key = session.user_key
        if user_key in self._user_sessions:
            self._user_sessions[user_key] = [
                s for s in self._user_sessions[user_key] if s != session_id
            ]
        if self._active.get(user_key) == session_id:
            remaining = self.list_sessions(user_key)
            if remaining:
                self._active[user_key] = remaining[0].id
            else:
                self._active.pop(user_key, None)
        self._save()
        return True

    def delete_sessions(self, session_ids: list[str]) -> int:
        """Delete multiple sessions. Returns count of successfully deleted."""
        count = 0
        for sid in session_ids:
            if self.delete_session(sid):
                count += 1
        return count

    def find_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def session_count(self, user_key: str) -> int:
        return len(self._user_sessions.get(user_key, []))

    # ── internal ──────────────────────────────────────────────────────────

    def _create_session(self, user_key: str, name: str = "") -> Session:
        self._counter += 1
        session = Session(id=str(uuid.uuid4()), user_key=user_key, name=name)
        self._sessions[session.id] = session
        self._user_sessions.setdefault(user_key, []).append(session.id)
        self._active[user_key] = session.id
        return session

    def _save(self) -> None:
        if not self._store_path:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        sessions_data: dict[str, Any] = {
            sid: {
                "id": s.id,
                "user_key": s.user_key,
                "agent_session_id": s.agent_session_id,
                "agent_type": s.agent_type,
                "name": s.name,
                "history": [
                    {
                        "role": h.role,
                        "content": h.content,
                        "timestamp": h.timestamp.isoformat(),
                    }
                    for h in s.history
                ],
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for sid, s in self._sessions.items()
        }
        data: dict[str, Any] = {
            "version": 2,
            "sessions": sessions_data,
            "active_sessions": dict(self._active),
            "session_counter": self._counter,
        }
        self._store_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if not self._store_path:
            return
        try:
            raw: dict[str, Any] = json.loads(self._store_path.read_text())
        except Exception:
            return
        if "version" not in raw:
            self._migrate_v1(raw)
            self._save()
            return
        self._counter = raw.get("session_counter", 0)
        self._active = raw.get("active_sessions", {})
        for sid, sd in raw.get("sessions", {}).items():
            history = [
                HistoryEntry(
                    role=h["role"],
                    content=h["content"],
                    timestamp=datetime.fromisoformat(h["timestamp"]),
                )
                for h in sd.get("history", [])
            ]
            agent_type = sd.get("agent_type", "gemini")
            session = Session(
                id=sd["id"],
                user_key=sd.get("user_key", ""),
                agent_session_id=sd.get("agent_session_id", ""),
                agent_type=agent_type
                if agent_type in ("gemini", "claude")
                else "gemini",
                name=sd.get("name", ""),
                history=history,
                created_at=datetime.fromisoformat(sd["created_at"]),
                updated_at=datetime.fromisoformat(sd["updated_at"]),
            )
            self._sessions[sid] = session
            if session.user_key:
                ulist = self._user_sessions.setdefault(session.user_key, [])
                if sid not in ulist:
                    ulist.append(sid)

    def _migrate_v1(self, raw: dict[str, Any]) -> None:
        """Migrate v1 format (user_key → session dict) to v2."""
        for user_key, sd in raw.items():
            if not isinstance(sd, dict) or "id" not in sd:
                continue
            session = Session(
                id=sd["id"],
                user_key=user_key,
                agent_session_id=sd.get("agent_session_id", ""),
                created_at=datetime.fromisoformat(sd["created_at"]),
                updated_at=datetime.fromisoformat(sd["updated_at"]),
            )
            self._sessions[session.id] = session
            self._user_sessions[user_key] = [session.id]
            self._active[user_key] = session.id
            self._counter += 1
