"""Gemini Commands: load .toml files from .gemini/commands/."""

import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

__all__ = ["CommandLoader", "GeminiCommand"]


@dataclass
class GeminiCommand:
    name: str  # e.g. "review" or "git-commit" (`:` replaced with `-`)
    description: str
    prompt: str
    source_path: Path


class CommandLoader:
    """Loads GeminiCommand objects from <work_dir>/.gemini/commands/*.toml."""

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir
        self._commands: dict[str, GeminiCommand] = {}

    def load(self) -> int:
        """Scan .gemini/commands/ and load all .toml files. Returns count loaded."""
        commands_dir = self._work_dir / ".gemini" / "commands"
        logger.info(f"loading commands from {commands_dir!s}")
        if not commands_dir.exists():
            return 0
        count = 0
        for toml_file in sorted(commands_dir.rglob("*.toml")):
            try:
                cmd = self._parse_command(toml_file, commands_dir)
                self._commands[cmd.name.lower()] = cmd
                logger.info(f"loaded command '{cmd.name}'")
                count += 1
            except Exception as exc:
                logger.warning(f"failed to load command from {toml_file!s}", error=exc)
        return count

    def reload(self) -> int:
        self._commands.clear()
        return self.load()

    def _parse_command(self, file: Path, base_dir: Path) -> GeminiCommand:
        rel = file.relative_to(base_dir)
        # git/commit.toml → "git_commit"  (only [a-z0-9_] allowed in Telegram commands)
        raw_name = str(rel.with_suffix("")).replace("/", ":").replace("\\", ":")
        name = raw_name.replace(":", "_")

        with file.open("rb") as f:
            data = tomllib.load(f)

        prompt = data.get("prompt", "").strip()
        if not prompt:
            msg = "missing 'prompt' field"
            raise ValueError(msg)

        description = data.get("description", f"Command: {name}")
        return GeminiCommand(
            name=name, description=description, prompt=prompt, source_path=file
        )

    def get(self, name: str) -> GeminiCommand | None:
        return self._commands.get(name.lower())

    def list_all(self) -> list[GeminiCommand]:
        return sorted(self._commands.values(), key=lambda c: c.name)

    async def expand_prompt(self, command: GeminiCommand, args: str) -> str:
        """Expand prompt syntax: @{file}, {{args}}, !{cmd}."""
        prompt = self._inject_files(command.prompt)
        if "{{args}}" in prompt:
            prompt = prompt.replace("{{args}}", args)
        elif args:
            prompt = prompt + "\n\n" + args
        return self._execute_shell_commands(prompt)

    def _inject_files(self, prompt: str) -> str:
        """Replace @{filepath} with file content."""

        def replace_file(match: re.Match[str]) -> str:
            filepath = match.group(1).strip()
            full_path = self._work_dir / filepath
            if not full_path.exists():
                return f"[File not found: {filepath}]"
            try:
                return full_path.read_text()
            except Exception as exc:
                return f"[Error reading {filepath}: {exc}]"

        return re.sub(r"@\{([^}]+)\}", replace_file, prompt)

    def _execute_shell_commands(self, prompt: str) -> str:
        """Replace !{cmd} with shell command output."""

        def replace_cmd(match: re.Match[str]) -> str:
            cmd = match.group(1).strip()
            try:
                result = subprocess.run(  # noqa: S602
                    cmd,
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self._work_dir,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr: {result.stderr}]"
                if result.returncode != 0:
                    output += f"\n[exit code: {result.returncode}]"
                return output
            except Exception as exc:
                return f"[Command failed: {exc}]"

        return re.sub(r"!\{([^}]+)\}", replace_cmd, prompt)
