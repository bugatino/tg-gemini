"""Claude Code CLI subprocess wrapper with bidirectional JSON streaming."""

import asyncio
import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from tg_gemini.models import Event, EventType, FileAttachment, ImageAttachment

__all__ = ["ClaudeAgent", "ClaudeSession"]


@dataclass
class _SessionInfo:
    """Session info parsed from Claude Code session files."""

    id: str
    summary: str
    message_count: int


def _normalize_mode(raw: str) -> str:
    """Map user-friendly mode aliases to Claude CLI values."""
    match raw.lower().strip():
        case "acceptedits" | "accept-edits" | "accept_edits" | "edit":
            return "acceptEdits"
        case "plan":
            return "plan"
        case (
            "bypasspermissions"
            | "bypass-permissions"
            | "bypass_permissions"
            | "yolo"
            | "auto"
        ):
            return "bypassPermissions"
        case "dontask" | "dont-ask" | "dont_ask":
            return "dontAsk"
        case _:
            return "default"


def _format_tool_params(tool_name: str, params: dict[str, Any]) -> str:
    """Extract human-readable summary from tool parameters (mirrors gemini.py logic)."""
    if not params:
        return ""

    match tool_name:
        case "Read":
            fp = params.get("file_path") or params.get("path", "")
            if fp:
                return str(fp)
        case "Write":
            fp = params.get("file_path") or params.get("path", "")
            if fp:
                content = params.get("content", "")
                if content:
                    return f"`{fp}`\n```\n{content}\n```"
                return str(fp)
        case "Edit":
            fp = params.get("file_path") or params.get("path", "")
            if fp:
                old = params.get("old_str") or params.get("old_string", "")
                new = params.get("new_str") or params.get("new_string", "")
                if old or new:
                    diff = _compute_line_diff(str(old), str(new))
                    if diff:
                        return f"`{fp}`\n```diff\n{diff}\n```"
                return str(fp)
        case "Bash" | "Shell":
            if cmd := params.get("command"):
                return f"```bash\n{cmd}\n```"
        case "Grep":
            if p := params.get("pattern"):
                return str(p)
        case "Glob":
            p = params.get("pattern") or params.get("glob_pattern", "")
            if p:
                return str(p)
        case "WebSearch":
            if q := params.get("query"):
                return str(q)
        case "WebFetch":
            if u := params.get("url"):
                return str(u)
        case "Task":
            if t := params.get("task"):
                return str(t)
        case "AskUserQuestion":
            questions = params.get("questions", [])
            if questions and isinstance(questions, list) and len(questions) > 0:
                q0 = questions[0]
                if isinstance(q0, dict) and (question := q0.get("question")):
                    return str(question)

    # Fallback: key: value pairs
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: {json.dumps(v)}")
    return ", ".join(parts)


def _compute_line_diff(old: str, new: str) -> str:
    """Compute a minimal unified-style diff between old and new text."""
    old_lines = old.split("\n")
    new_lines = new.split("\n")

    # Find common prefix
    prefix_len = 0
    min_len = min(len(old_lines), len(new_lines))
    while prefix_len < min_len and old_lines[prefix_len] == new_lines[prefix_len]:
        prefix_len += 1

    # Find common suffix
    suffix_len = 0
    while (
        suffix_len < len(old_lines) - prefix_len
        and suffix_len < len(new_lines) - prefix_len
        and old_lines[len(old_lines) - 1 - suffix_len]
        == new_lines[len(new_lines) - 1 - suffix_len]
    ):
        suffix_len += 1

    if prefix_len + suffix_len >= len(old_lines) and prefix_len + suffix_len >= len(
        new_lines
    ):
        return ""

    if prefix_len == 0 and suffix_len == 0:
        lines = [f"- {ln}" for ln in old_lines] + [f"+ {ln}" for ln in new_lines]
        return "\n".join(lines)

    parts = []
    ctx_start = max(0, prefix_len - 1)
    if ctx_start > 0:
        parts.append("  ...")
    parts.extend(f"  {old_lines[i]}" for i in range(ctx_start, prefix_len))
    parts.extend(
        f"- {old_lines[i]}" for i in range(prefix_len, len(old_lines) - suffix_len)
    )
    parts.extend(
        f"+ {new_lines[i]}" for i in range(prefix_len, len(new_lines) - suffix_len)
    )
    suff_start = len(old_lines) - suffix_len
    suff_end = min(suff_start + 1, len(old_lines))
    parts.extend(f"  {old_lines[i]}" for i in range(suff_start, suff_end))
    if suff_end < len(old_lines):
        parts.append("  ...")

    return "\n".join(parts)


class ClaudeSession:
    """Manages a long-running Claude Code subprocess with bidirectional JSON streaming.

    Claude Code uses --input-format stream-json and --permission-prompt-tool stdio
    for bidirectional communication. User messages are written to stdin as JSON.
    Server events are read from stdout as JSONL.
    """

    def __init__(
        self,
        cmd: str,
        work_dir: str,
        model: str,
        mode: str,
        allowed_tools: list[str] | None,
        disallowed_tools: list[str] | None,
        timeout_mins: int,
        resume_id: str = "",
    ) -> None:
        self._cmd = cmd
        self._work_dir = work_dir
        self._model = model
        self._mode = _normalize_mode(mode)
        self._allowed_tools = allowed_tools or []
        self._disallowed_tools = disallowed_tools or []
        self._timeout_secs = timeout_mins * 60 if timeout_mins > 0 else None
        self._resume_id = resume_id
        self._session_id = resume_id
        self._events: asyncio.Queue[Event] = asyncio.Queue(maxsize=256)
        self._alive = True
        self._temp_files: list[str] = []
        self._proc: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._stdin: asyncio.StreamWriter | None = None

    @property
    def events(self) -> asyncio.Queue[Event]:
        return self._events

    @property
    def current_session_id(self) -> str:
        return self._session_id

    @property
    def alive(self) -> bool:
        return self._alive

    async def send(
        self,
        prompt: str,
        images: list[ImageAttachment] | None = None,
        files: list[FileAttachment] | None = None,
    ) -> None:
        """Send a user message to the Claude Code subprocess via stdin."""
        if not self._alive:
            raise RuntimeError("Session is closed")

        # Ensure subprocess is running
        if self._proc is None:
            await self._start_subprocess()

        # Save images/files to temp for @file references
        image_refs: list[str] = []
        file_refs: list[str] = []
        tmp_dir = tempfile.gettempdir()

        for i, img in enumerate(images or []):
            ext = {
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
            }.get(img.mime_type, ".png")
            fpath = Path(tmp_dir) / f"tg-gemini-img-{i}{ext}"
            fpath.write_bytes(img.data)
            image_refs.append(str(fpath))
            self._temp_files.append(str(fpath))

        for i, f in enumerate(files or []):
            fname = Path(f.file_name).name or f"tg-gemini-file-{i}"
            fpath = Path(tmp_dir) / fname
            fpath.write_bytes(f.data)
            file_refs.append(str(fpath))
            self._temp_files.append(str(fpath))

        # Build content array for multimodal message
        content: list[dict[str, Any]] = []

        # Add image refs to prompt
        if image_refs:
            content.append({"type": "text", "text": " ".join(image_refs)})

        if file_refs:
            content.append({"type": "text", "text": " ".join(file_refs)})

        if content:
            content.append({"type": "text", "text": prompt})
        else:
            content.append({"type": "text", "text": prompt})

        msg: dict[str, Any] = {
            "type": "user",
            "message": {
                "role": "user",
                "content": content if len(content) > 1 else prompt,
            },
        }

        # If simple text-only, send as plain string content
        if len(content) == 1 and isinstance(content[0]["text"], str):
            msg = {"type": "user", "message": {"role": "user", "content": prompt}}

        await self._write_json(msg)

    async def respond_permission(
        self, request_id: str, allow: bool, message: str = ""
    ) -> None:
        """Respond to a permission request by writing control_response to stdin."""
        if not self._alive or self._stdin is None:
            return

        response: dict[str, Any] = {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": {"behavior": "allow" if allow else "deny"},
            },
        }
        if not allow and message:
            response["response"]["response"]["message"] = message

        await self._write_json(response)

    async def _write_json(self, data: dict[str, Any]) -> None:
        """Write JSON data to subprocess stdin."""
        if self._stdin is None:
            return
        line = json.dumps(data).encode() + b"\n"
        self._stdin.write(line)
        await self._stdin.drain()

    async def _start_subprocess(self) -> None:
        """Launch the Claude Code subprocess."""
        args: list[str] = [
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--permission-prompt-tool",
            "stdio",
        ]

        if self._mode and self._mode != "default":
            args += ["--permission-mode", self._mode]

        if self._resume_id:
            if self._resume_id == "_continue":
                # Special case: --continue --fork-session for resuming last session
                args += ["--continue", "--fork-session"]
            else:
                args += ["--resume", self._resume_id]

        if self._model:
            args += ["--model", self._model]

        if self._allowed_tools:
            args += ["--allowedTools", ",".join(self._allowed_tools)]

        if self._disallowed_tools:
            args += ["--disallowedTools", ",".join(self._disallowed_tools)]

        logger.debug("ClaudeSession: launching", args=args[:5])

        # limit=10 MiB: large tool_result JSON can exceed default 64 KiB
        self._proc = await asyncio.create_subprocess_exec(
            self._cmd,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            cwd=self._work_dir,
            env=os.environ.copy(),
            limit=10 * 1024 * 1024,
        )
        self._stdin = self._proc.stdin

        self._read_task = asyncio.create_task(self._read_loop(self._proc))
        self._read_task.add_done_callback(
            lambda t: (
                logger.error("ClaudeSession: read loop crashed", exc_info=t.exception())
                if not t.cancelled() and t.exception()
                else None
            )
        )

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        """Read JSONL output from Claude Code stdout and emit events."""
        assert proc.stdout is not None
        assert proc.stderr is not None

        # Drain stderr concurrently to prevent pipe-buffer deadlock
        stderr_chunks: list[str] = []

        async def _drain_stderr() -> None:
            assert proc.stderr is not None
            async for line_bytes in proc.stderr:
                stderr_chunks.append(line_bytes.decode(errors="replace"))  # noqa: PERF401

        stderr_task = asyncio.create_task(_drain_stderr())

        try:
            coro = self._stream_stdout(proc.stdout)
            if self._timeout_secs:
                await asyncio.wait_for(coro, timeout=self._timeout_secs)
            else:
                await coro
        except TimeoutError:
            logger.warning(
                "ClaudeSession: timeout, killing process", timeout=self._timeout_secs
            )
            proc.kill()
            await self._events.put(
                Event(
                    type=EventType.ERROR, error=TimeoutError("Claude process timed out")
                )
            )
        finally:
            for f in self._temp_files:
                with contextlib.suppress(OSError):
                    Path(f).unlink()
            self._temp_files.clear()

            await proc.wait()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

            self._alive = False
            if self._stdin:
                with contextlib.suppress(OSError):
                    self._stdin.close()
                self._stdin = None

            if proc.returncode != 0:
                stderr_msg = "".join(stderr_chunks).strip()
                if stderr_msg:
                    logger.error("ClaudeSession: process failed", stderr=stderr_msg)
                    await self._events.put(
                        Event(type=EventType.ERROR, error=RuntimeError(stderr_msg))
                    )

    async def _stream_stdout(self, stdout: asyncio.StreamReader) -> None:
        """Read and parse JSONL lines from stdout."""
        try:
            async for line_bytes in stdout:
                line = line_bytes.decode(errors="replace").rstrip()
                if not line:
                    continue
                logger.debug("ClaudeSession: raw line", line=line[:200])
                self._parse_line(line)
        except ValueError:
            # LimitOverrunError wrapper
            logger.error("ClaudeSession: stdout line too long")

    def _parse_line(self, line: str) -> None:
        """Parse a JSONL line, extracting all JSON objects from it."""
        remaining = line
        while True:
            start = remaining.find("{")
            if start == -1:
                break
            remaining = remaining[start:]
            try:
                obj, end = self._decode_first_json(remaining)
                self._handle_event(obj)
                remaining = remaining[end:]
            except (json.JSONDecodeError, ValueError):
                remaining = remaining[1:]

    @staticmethod
    def _decode_first_json(text: str) -> tuple[dict[str, Any], int]:
        """Decode the first complete JSON object from text."""
        decoder = json.JSONDecoder()
        obj, end = decoder.raw_decode(text)
        if not isinstance(obj, dict):
            raise TypeError("not a dict")
        return obj, end

    def _handle_event(self, raw: dict[str, Any]) -> None:
        event_type = raw.get("type", "")
        match event_type:
            case "system":
                self._handle_system(raw)
            case "assistant":
                self._handle_assistant(raw)
            case "user":
                pass  # User messages echoed back - skip
            case "result":
                self._handle_result(raw)
            case "control_request":
                self._handle_control_request(raw)
            case "control_cancel_request":
                request_id = raw.get("request_id", "")
                logger.debug(
                    "ClaudeSession: permission cancelled", request_id=request_id
                )
            case _:
                logger.debug("ClaudeSession: unhandled event type", type=event_type)

    def _handle_system(self, raw: dict[str, Any]) -> None:
        """Handle system event - stores session_id."""
        if sid := raw.get("session_id"):
            self._session_id = str(sid)
            logger.debug("ClaudeSession: session_id", session_id=sid)
            self._events.put_nowait(Event(type=EventType.TEXT, session_id=str(sid)))

    def _handle_assistant(self, raw: dict[str, Any]) -> None:
        """Handle assistant message - extract text, thinking, and tool_use."""
        msg = raw.get("message", {})
        if not isinstance(msg, dict):
            return

        content_arr = msg.get("content", [])
        if not isinstance(content_arr, list):
            # Plain text content
            if text := msg.get("content"):
                self._events.put_nowait(Event(type=EventType.TEXT, content=str(text)))
            return

        for item in content_arr:
            if not isinstance(item, dict):
                continue
            content_type = item.get("type", "")
            match content_type:
                case "text":
                    if text := item.get("text"):
                        self._events.put_nowait(
                            Event(type=EventType.TEXT, content=str(text))
                        )
                case "thinking":
                    if thinking := item.get("thinking"):
                        self._events.put_nowait(
                            Event(type=EventType.THINKING, content=str(thinking))
                        )
                case "tool_use":
                    tool_name = item.get("name", "")
                    if tool_name == "AskUserQuestion":
                        continue  # Skip AskUserQuestion tool use display
                    tool_input = _format_tool_params(tool_name, item.get("input", {}))
                    self._events.put_nowait(
                        Event(
                            type=EventType.TOOL_USE,
                            tool_name=tool_name,
                            tool_input=tool_input,
                        )
                    )

    def _handle_result(self, raw: dict[str, Any]) -> None:
        """Handle result event - final response."""
        content = raw.get("result", "")
        if sid := raw.get("session_id"):
            self._session_id = str(sid)
        self._events.put_nowait(
            Event(type=EventType.RESULT, content=str(content), done=True)
        )

    def _handle_control_request(self, raw: dict[str, Any]) -> None:
        """Handle permission request - emit PERMISSION_REQUEST event."""
        request_id = raw.get("request_id", "")
        request = raw.get("request", {})
        if not isinstance(request, dict):
            return

        subtype = request.get("subtype", "")
        if subtype != "can_use_tool":
            logger.debug(
                "ClaudeSession: unknown control request subtype", subtype=subtype
            )
            return

        tool_name = request.get("tool_name", "")
        tool_input = _format_tool_params(tool_name, request.get("input", {}))

        self._events.put_nowait(
            Event(
                type=EventType.PERMISSION_REQUEST,
                request_id=request_id,
                tool_name=tool_name,
                tool_input=tool_input,
            )
        )

    async def kill(self) -> None:
        """Send SIGTERM then SIGKILL to the subprocess if still running."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except TimeoutError:
            proc.kill()

    async def close(self) -> None:
        self._alive = False
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task
            self._read_task = None
        if self._stdin:
            with contextlib.suppress(OSError):
                self._stdin.close()
            self._stdin = None
        await self.kill()


class ClaudeAgent:
    """Factory for ClaudeSession instances."""

    def __init__(
        self,
        work_dir: str = ".",
        model: str = "",
        mode: str = "default",
        cmd: str = "claude",
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        timeout_mins: int = 0,
    ) -> None:
        self._work_dir = work_dir
        self._model = model
        self._mode = _normalize_mode(mode)
        self._cmd = cmd
        self._allowed_tools = allowed_tools or []
        self._disallowed_tools = disallowed_tools or []
        self._timeout_mins = timeout_mins

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = _normalize_mode(value)

    def start_session(self, resume_id: str = "") -> "ClaudeSession":
        """Create a new ClaudeSession, optionally resuming by session ID."""
        return ClaudeSession(
            cmd=self._cmd,
            work_dir=self._work_dir,
            model=self._model,
            mode=self._mode,
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            timeout_mins=self._timeout_mins,
            resume_id=resume_id,
        )

    def available_models(self) -> list:
        """Return a list of known Claude models."""
        return [
            {"name": "sonnet", "desc": "Claude Sonnet 4 (balanced)"},
            {"name": "opus", "desc": "Claude Opus 4 (most capable)"},
            {"name": "haiku", "desc": "Claude Haiku 3.5 (fastest)"},
        ]

    async def list_sessions(self) -> list[_SessionInfo]:
        """List sessions by scanning ~/.claude/projects/ for session JSONL files."""
        home = Path.home()
        projects_dir = home / ".claude" / "projects"
        if not projects_dir.exists():
            return []

        sessions: list[_SessionInfo] = []
        abs_work_dir = Path(self._work_dir).resolve()

        # Find project dir for this work_dir
        project_key = str(abs_work_dir).replace(os.sep, "-")
        candidates = [
            project_key,
            project_key.replace(":", "-").replace("/", "-").replace("\\", "-"),
        ]

        project_dir = None
        for candidate in candidates:
            candidate_dir = projects_dir / candidate
            if candidate_dir.exists():
                project_dir = candidate_dir
                break

        if project_dir is None and projects_dir.is_dir():
            # Fallback: scan all dirs
            for entry in projects_dir.iterdir():
                if entry.is_dir() and entry.name == project_key:
                    project_dir = entry
                    break

        if project_dir is None or not project_dir.exists():
            return []

        for entry in project_dir.iterdir():
            if entry.is_file() and entry.suffix == ".jsonl":
                session_id = entry.stem
                summary, msg_count = self._scan_session_meta(entry)
                sessions.append(
                    _SessionInfo(
                        id=session_id, summary=summary, message_count=msg_count
                    )
                )

        sessions.sort(key=lambda s: s.message_count, reverse=True)
        return sessions

    def _scan_session_meta(self, path: Path) -> tuple[str, int]:
        """Read a session JSONL file to extract summary and message count."""
        try:
            content = path.read_text(errors="replace")
        except OSError:
            return "", 0

        lines = content.splitlines()
        summary = ""
        msg_count = 0

        for line in lines[-100:]:  # Check last 100 lines for performance
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") in ("user", "assistant"):
                    msg_count += 1
                    if obj.get("type") == "user":
                        msg_content = obj.get("message", {})
                        if isinstance(msg_content, dict):
                            text = msg_content.get("content", "")
                            if isinstance(text, str) and text:
                                summary = text[:40]
                        elif isinstance(msg_content, str):
                            summary = msg_content[:40]
            except (json.JSONDecodeError, ValueError):
                continue

        if len(summary) == 40:
            summary += "…"
        return summary.strip(), msg_count

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by removing its JSONL file."""
        home = Path.home()
        projects_dir = home / ".claude" / "projects"
        if not projects_dir.exists():
            return False

        abs_work_dir = Path(self._work_dir).resolve()
        project_key = str(abs_work_dir).replace(os.sep, "-")
        candidates = [
            project_key,
            project_key.replace(":", "-").replace("/", "-").replace("\\", "-"),
        ]

        for candidate in candidates:
            session_file = projects_dir / candidate / f"{session_id}.jsonl"
            if session_file.exists():
                try:
                    session_file.unlink()
                    return True
                except OSError:
                    return False

        return False
