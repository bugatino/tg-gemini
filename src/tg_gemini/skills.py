"""Skills system: load SKILL.md files from skill directories."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

__all__ = ["Skill", "SkillRegistry"]


@dataclass
class Skill:
    name: str  # directory name, e.g. "review"
    display_name: str  # frontmatter `name` or directory name
    description: str  # frontmatter `description` or first line of body
    prompt: str  # SKILL.md body
    source_dir: Path


class SkillRegistry:
    """Scans skill directories and loads SKILL.md files.

    Auto-loads from <work_dir>/.gemini/skills/ (like CommandLoader).
    Additional directories can be added via add_directory().
    """

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir
        self._extra_dirs: list[Path] = []
        self._skills: dict[str, Skill] = {}

    def add_directory(self, path: Path) -> None:
        """Add an additional skill directory (e.g., from config)."""
        if path not in self._extra_dirs:
            self._extra_dirs.append(path)

    def load(self) -> int:
        """Scan skill_dirs and load all SKILL.md files. Returns count loaded."""
        count = 0
        # Default: <work_dir>/.gemini/skills/
        default_dir = self._work_dir / ".gemini" / "skills"
        dirs = (
            [default_dir, *self._extra_dirs]
            if default_dir.exists()
            else self._extra_dirs
        )

        for dir_path in dirs:
            if not dir_path.exists():
                continue
            logger.info(f"loading skills from {dir_path!s}")
            for subdir in sorted(dir_path.iterdir()):
                if not subdir.is_dir():
                    continue
                skill_file = subdir / "SKILL.md"
                if skill_file.exists():
                    try:
                        skill = self._parse_skill(skill_file, subdir.name)
                        self._skills[skill.name.lower()] = skill
                        logger.info(f"loaded skill '{skill.name}'")
                        count += 1
                    except Exception as exc:
                        logger.warning(
                            f"failed to load skill from {skill_file!s}", error=exc
                        )
        return count

    def invalidate(self) -> None:
        self._skills.clear()

    def _parse_skill(self, file: Path, name: str) -> Skill:
        content = file.read_text()
        frontmatter, body = self._extract_frontmatter(content)
        first_line = body.strip().split("\n")[0][:80] if body.strip() else name
        return Skill(
            name=name,
            display_name=frontmatter.get("name", name),
            description=frontmatter.get("description", first_line),
            prompt=body.strip(),
            source_dir=file.parent,
        )

    @staticmethod
    def _extract_frontmatter(content: str) -> tuple[dict[str, str], str]:
        """Extract YAML frontmatter. Returns (frontmatter_dict, body)."""
        if not content.startswith("---"):
            return {}, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        try:
            fm = yaml.safe_load(parts[1]) or {}
            return fm, parts[2]
        except yaml.YAMLError:
            return {}, content

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name, ignoring case and treating `-`/`_` as equivalent."""
        normalized = name.lower().replace("-", "_")
        for key, skill in self._skills.items():
            if key.replace("-", "_") == normalized:
                return skill
        return None

    def list_all(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    @staticmethod
    def build_invocation_prompt(skill: Skill, args: str) -> str:
        """Build the full prompt for invoking a skill."""
        parts = [
            f"The user wants you to execute the skill: {skill.display_name or skill.name}",
            "",
            f"## Description: {skill.description}",
            "",
            "## Skill Instructions:",
            skill.prompt,
        ]
        if args:
            parts.extend(["", "## User Arguments:", args])
        parts.extend(["", "Please follow the skill instructions to complete the task."])
        return "\n".join(parts)
