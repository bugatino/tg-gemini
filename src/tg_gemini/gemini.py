"""Gemini CLI subprocess wrapper with JSONL stream-json parsing."""

import asyncio
import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from loguru import logger

from tg_gemini.config import GeminiMode
from tg_gemini.models import (
    Event,
    EventType,
    FileAttachment,
    ImageAttachment,
    ModelOption,
)

__all__ = ["GeminiAgent", "GeminiSession", "SessionInfo"]


@dataclass(frozen=True)
class SessionInfo:
    """Session info from gemini --list-sessions output."""

    index: int
    title: str
    time: str
    session_id: str


_KNOWN_MODELS: list[ModelOption] = [
    ModelOption(name="gemini-3.1-pro-preview", desc="Gemini 3.1 Pro Preview"),
    ModelOption(name="gemini-3-flash-preview", desc="Gemini 3 Flash Preview"),
    ModelOption(name="gemini-2.5-pro", desc="Gemini 2.5 Pro"),
    ModelOption(name="gemini-2.5-flash", desc="Gemini 2.5 Flash"),
    ModelOption(name="gemini-2.5-flash-lite", desc="Gemini 2.5 Flash Lite"),
]


def _normalize_mode(raw: str) -> GeminiMode:
    match raw.lower().strip():
        case "yolo" | "auto" | "force":
            return "yolo"
        case "auto_edit" | "autoedit" | "edit":
            return "auto_edit"
        case "plan":
            return "plan"
        case _:
            return "default"


def _format_tool_params(tool_name: str, params: dict[str, Any]) -> str:
    """Extract human-readable summary from tool parameters."""
    if not params:
        return ""

    match tool_name:
        case "shell" | "run_shell_command" | "Bash":
            if cmd := params.get("command"):
                return f"```bash\n{cmd}\n```"
        case "write_file" | "WriteFile":
            fp = params.get("file_path") or params.get("path", "")
            if fp:
                content = params.get("content", "")
                if content:
                    return f"`{fp}`\n```\n{content}\n```"
                return str(fp)
        case "replace" | "ReplaceInFile":
            fp = params.get("file_path") or params.get("path", "")
            if fp:
                old = params.get("old_string") or params.get("old_str", "")
                new = params.get("new_string") or params.get("new_str", "")
                if old or new:
                    diff = _compute_line_diff(str(old), str(new))
                    return f"`{fp}`\n```diff\n{diff}\n```"
                return str(fp)
        case "read_file" | "ReadFile":
            p = params.get("file_path") or params.get("path", "")
            if p:
                return str(p)
        case "list_directory" | "ListDirectory":
            p = (
                params.get("dir_path")
                or params.get("path")
                or params.get("directory", "")
            )
            if p:
                return str(p)
        case "web_fetch" | "WebFetch":
            p = params.get("prompt") or params.get("url", "")
            if p:
                return str(p)
        case "google_web_search" | "GoogleWebSearch":
            if q := params.get("query"):
                return str(q)
        case "activate_skill":
            if n := params.get("name"):
                return str(n)
        case "search_code" | "SearchCode" | "Glob" | "glob" | "Grep" | "grep_search":
            p = params.get("query") or params.get("pattern", "")
            if p:
                return str(p)
        case "save_memory":
            if f := params.get("fact"):
                return str(f)
        case "ask_user":
            questions = params.get("questions", [])
            if questions and isinstance(questions, list) and len(questions) > 0:
                q0 = questions[0]
                if isinstance(q0, dict) and (question := q0.get("question")):
                    return str(question)
        case "enter_plan_mode":
            if r := params.get("reason"):
                return str(r)
        case "exit_plan_mode":
            if p := params.get("plan_path"):
                return str(p)

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

    # No changes
    if prefix_len + suffix_len >= len(old_lines) and prefix_len + suffix_len >= len(
        new_lines
    ):
        return ""

    # Everything differs
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


class GeminiSession:
    """Manages a single turn with Gemini CLI via --output-format stream-json."""

    def __init__(
        self,
        cmd: str,
        work_dir: str,
        model: str,
        mode: GeminiMode,
        api_key: str,
        timeout_mins: int,
        resume_id: str = "",
    ) -> None:
        self._cmd = cmd
        self._work_dir = work_dir
        self._model = model
        self._mode: GeminiMode = mode
        self._api_key = api_key
        self._timeout_secs = timeout_mins * 60 if timeout_mins > 0 else None
        self._session_id = resume_id
        self._events: asyncio.Queue[Event] = asyncio.Queue(maxsize=256)
        self._pending_msgs: list[str] = []
        self._alive = True
        self._temp_files: list[str] = []
        self._proc: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task[None] | None = None

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
        """Launch gemini CLI process and start reading JSONL events."""
        if not self._alive:
            raise RuntimeError("Session is closed")

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

        # Build CLI args
        args: list[str] = ["--output-format", "stream-json"]

        match self._mode:
            case "yolo":
                args += ["-y"]
            case "auto_edit":
                args += ["--approval-mode", "auto_edit"]
            case "plan":
                args += ["--approval-mode", "plan"]

        if self._session_id:
            args += ["--resume", self._session_id]
        if self._model:
            args += ["-m", self._model]

        full_prompt = prompt
        if image_refs:
            full_prompt = " ".join(image_refs) + " " + full_prompt
        if file_refs:
            full_prompt = " ".join(file_refs) + " " + full_prompt

        args += ["-p", full_prompt]

        env = os.environ.copy()
        if self._api_key:
            env["GEMINI_API_KEY"] = self._api_key

        logger.debug("GeminiSession: launching", args=args[:5])

        # limit=10 MiB: Gemini tool_result JSON lines can be very large;
        # the default 64 KiB StreamReader limit causes LimitOverrunError.
        proc = await asyncio.create_subprocess_exec(
            self._cmd,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._work_dir,
            env=env,
            limit=10 * 1024 * 1024,
        )
        self._proc = proc

        self._read_task = asyncio.create_task(self._read_loop(proc))
        self._read_task.add_done_callback(
            lambda t: (
                logger.error("GeminiSession: read loop crashed", exc_info=t.exception())
                if not t.cancelled() and t.exception()
                else None
            )
        )

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        """Read JSONL output from the Gemini process and emit events."""
        assert proc.stdout is not None
        assert proc.stderr is not None

        # Drain stderr concurrently to prevent pipe-buffer deadlock:
        # if stderr fills up (~64 KB), the process blocks on write and
        # can't produce more stdout, causing a silent hang.
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
                "GeminiSession: timeout, killing process", timeout=self._timeout_secs
            )
            proc.kill()
            await self._events.put(
                Event(
                    type=EventType.ERROR, error=TimeoutError("Gemini process timed out")
                )
            )
        finally:
            for f in self._temp_files:
                with contextlib.suppress(OSError):
                    Path(f).unlink()
            self._temp_files.clear()

            await proc.wait()
            # stderr pipe closes after process exit; let drain finish
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
            if proc.returncode != 0:
                stderr_msg = "".join(stderr_chunks).strip()
                if stderr_msg:
                    logger.error("GeminiSession: process failed", stderr=stderr_msg)
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
                logger.debug("GeminiSession: raw line", line=line[:200])
                self._parse_line(line)
        except ValueError as exc:
            # LimitOverrunError is wrapped as ValueError by StreamReader.readline
            logger.error("GeminiSession: stdout line too long", error=exc)

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
                # Skip one char and try from next {
                remaining = remaining[1:]

    @staticmethod
    def _decode_first_json(text: str) -> tuple[dict[str, Any], int]:
        """Decode the first complete JSON object from text, return (obj, end_index)."""
        decoder = json.JSONDecoder()
        obj, end = decoder.raw_decode(text)
        if not isinstance(obj, dict):
            raise TypeError("not a dict")
        return obj, end

    def _handle_event(self, raw: dict[str, Any]) -> None:
        event_type = raw.get("type", "")
        match event_type:
            case "init":
                self._handle_init(raw)
            case "message":
                self._handle_message(raw)
            case "tool_use":
                self._handle_tool_use(raw)
            case "tool_result":
                self._handle_tool_result(raw)
            case "error":
                self._handle_error(raw)
            case "result":
                self._handle_result(raw)
            case _:
                logger.debug("GeminiSession: unhandled event type", type=event_type)

    def _handle_init(self, raw: dict[str, Any]) -> None:
        sid = raw.get("session_id", "")
        model = raw.get("model", "")
        if sid:
            self._session_id = str(sid)
            logger.debug("GeminiSession: init", session_id=sid, model=model)
            # Emit init as EventText with session_id and model info
            evt = Event(type=EventType.TEXT, session_id=str(sid), tool_name=str(model))
            self._events.put_nowait(evt)

    def _handle_message(self, raw: dict[str, Any]) -> None:
        role = raw.get("role", "")
        content = raw.get("content", "")
        if role == "user" or not content:
            return
        delta = raw.get("delta", False)
        # Strip [Thought: xxx] prefix from thinking content; emit as THINKING
        # event immediately so each thinking step appears as a separate message.
        # Format is either "[Thought: true]Some text" or "[Thought: actual thought]"
        if content.startswith("[Thought: "):
            inner = content.removeprefix("[Thought: ")
            if inner.endswith("]"):
                inner = inner.removesuffix("]")
            thought_text = inner.strip()
            if thought_text:
                self._events.put_nowait(
                    Event(type=EventType.THINKING, content=thought_text)
                )
            return
        if delta:
            self._events.put_nowait(Event(type=EventType.TEXT, content=str(content)))
        else:
            self._pending_msgs.append(str(content))

    def _handle_tool_use(self, raw: dict[str, Any]) -> None:
        self._flush_pending_as_thinking()
        tool_name = str(raw.get("tool_name", ""))
        params = raw.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        tool_input = _format_tool_params(tool_name, params)
        self._events.put_nowait(
            Event(type=EventType.TOOL_USE, tool_name=tool_name, tool_input=tool_input)
        )

    def _handle_tool_result(self, raw: dict[str, Any]) -> None:
        tool_id = str(raw.get("tool_id", ""))
        status = str(raw.get("status", ""))
        output = str(raw.get("output", ""))
        if status == "error":
            err_obj = raw.get("error")
            if isinstance(err_obj, dict):
                err_msg = str(err_obj.get("message", ""))
                if err_msg:
                    output = f"Error: {err_msg}"
        if output:
            truncated = output[:500] + "..." if len(output) > 500 else output
            self._events.put_nowait(
                Event(type=EventType.TOOL_RESULT, tool_name=tool_id, content=truncated)
            )

    def _handle_error(self, raw: dict[str, Any]) -> None:
        severity = str(raw.get("severity", "error"))
        message = str(raw.get("message", ""))
        if message:
            self._events.put_nowait(
                Event(
                    type=EventType.ERROR, error=RuntimeError(f"[{severity}] {message}")
                )
            )

    def _handle_result(self, raw: dict[str, Any]) -> None:
        self._flush_pending_as_text()
        status = str(raw.get("status", ""))
        sid = self._session_id
        stats = raw.get("stats") or {}
        if stats:
            logger.info("GeminiSession: usage", **{str(k): v for k, v in stats.items()})
        if status == "error":
            err_obj = raw.get("error")
            err_msg = ""
            if isinstance(err_obj, dict):
                err_msg = str(err_obj.get("message", ""))
            self._events.put_nowait(
                Event(
                    type=EventType.RESULT,
                    session_id=sid,
                    done=True,
                    error=RuntimeError(err_msg) if err_msg else None,
                    content=err_msg,
                )
            )
        else:
            self._events.put_nowait(
                Event(type=EventType.RESULT, session_id=sid, done=True)
            )

    def _flush_pending_as_thinking(self) -> None:
        if self._pending_msgs:
            text = "".join(self._pending_msgs)
            self._pending_msgs.clear()
            if text:
                self._events.put_nowait(Event(type=EventType.THINKING, content=text))

    def _flush_pending_as_text(self) -> None:
        if self._pending_msgs:
            text = "".join(self._pending_msgs)
            self._pending_msgs.clear()
            if text:
                self._events.put_nowait(Event(type=EventType.TEXT, content=text))

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
        await self.kill()


class GeminiAgent:
    """Factory for GeminiSession instances."""

    def __init__(
        self,
        work_dir: str = ".",
        model: str = "",
        mode: GeminiMode = "default",
        cmd: str = "gemini",
        api_key: str = "",
        timeout_mins: int = 0,
    ) -> None:
        self._work_dir = work_dir
        self._model = model
        self._mode: GeminiMode = _normalize_mode(mode)
        self._cmd = cmd
        self._api_key = api_key
        self._timeout_mins = timeout_mins

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @property
    def mode(self) -> GeminiMode:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = _normalize_mode(value)

    def start_session(self, resume_id: str = "") -> "GeminiSession":
        """Create a new GeminiSession, optionally resuming by session ID."""
        return GeminiSession(
            cmd=self._cmd,
            work_dir=self._work_dir,
            model=self._model,
            mode=self._mode,
            api_key=self._api_key,
            timeout_mins=self._timeout_mins,
            resume_id=resume_id,
        )

    def available_models(self) -> list[ModelOption]:
        """Return a hardcoded list of known Gemini models."""
        return list(_KNOWN_MODELS)

    async def list_sessions(self) -> list[SessionInfo]:
        """List sessions via `gemini --list-sessions` and parse output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cmd,
                "--list-sessions",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._work_dir,
            )
            stdout, _ = await proc.communicate()
            lines = stdout.decode(errors="replace").strip().splitlines()
        except Exception as e:
            logger.warning(
                "list_sessions: failed to run gemini --list-sessions", error=e
            )
            return []

        sessions: list[SessionInfo] = []
        for line in lines:
            m = re.match(r"\s*(\d+)\.\s+(.+?)\s+\((\d+)\s+\)", line)
            if not m:
                continue
            sessions.append(
                SessionInfo(
                    index=int(m.group(1)),
                    title=m.group(2).strip(),
                    time=m.group(3),
                    session_id=m.group(3),
                )
            )
        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session via `gemini --delete-session <id>`."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cmd,
                "--delete-session",
                session_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._work_dir,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception as e:
            logger.warning("delete_session: failed", session_id=session_id, error=e)
            return False

    async def run_stream(
        self,
        prompt: str,
        session_id: str | None,
        model: str | None,
        stop_event: asyncio.Event,
    ) -> "AsyncGenerator[Event, None]":
        """Stream events from a Gemini session with stop/interrupt support.

        Args:
            prompt: The user prompt to send to Gemini.
            session_id: Session ID to resume (None or "" for new session).
            model: Model name override for this session.
            stop_event: asyncio.Event to signal interruption.

        Yields:
            Event objects from the Gemini CLI stream.
        """
        session = GeminiSession(
            cmd=self._cmd,
            work_dir=self._work_dir,
            model=model or self._model,
            mode=self._mode,
            api_key=self._api_key,
            timeout_mins=self._timeout_mins,
            resume_id=session_id or "",
        )
        try:
            await session.send(prompt)
            while True:
                if stop_event.is_set():
                    logger.debug("run_stream: interrupted by stop_event")
                    break
                try:
                    event = await asyncio.wait_for(session.events.get(), timeout=0.5)
                except TimeoutError:
                    continue  # Re-check stop on next iteration
                except asyncio.CancelledError:
                    break
                yield event
                if event.type == EventType.RESULT:
                    break
        finally:
            await session.close()
