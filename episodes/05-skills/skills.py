"""
Skills

A skill is a folder under .skills/ holding a SKILL.md: a short front matter
(name, description) and a body of instructions. The agent sees every skill's
name and description in its system prompt from the start (skills_index), and
loads a body only when that description matches the work in front of it
(load_skill, the one tool here). So an unused skill costs one line of the
prompt, and a loaded one arrives as a tool result, like a file it has read.

Skills are agent infrastructure, not part of the project the agent works on:
they sit next to agent.py, outside initial/ and outside the sandbox, so the
file tools never see them and load_skill is the only way in.

See ../../README.md for context.
"""
from pathlib import Path

from tools import tool

SKILLS_DIR = Path(".skills")


def parse_skill(path: Path) -> tuple[dict, str]:
    """Split a SKILL.md into its front matter (the `key: value` lines between
    the two --- fences, as a dict) and its body."""
    _, front, body = path.read_text(encoding="utf-8").split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.strip()


def skills_index() -> str:
    """One line per skill, name and description, for the system prompt. The
    folder is named after the skill, so the name here is also what load_skill
    takes."""
    lines = []
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        meta, _ = parse_skill(skill_md)
        lines.append(f"- {skill_md.parent.name}: {meta['description']}")
    return "\n".join(lines)


@tool(
    "Load a skill's full instructions by name. The available skills are listed "
    "in your system prompt with a description of when each applies; load one "
    "when its description matches the work in front of you."
)
def load_skill(name: str) -> str:
    skills = {p.parent.name: p for p in SKILLS_DIR.glob("*/SKILL.md")}
    if name not in skills:
        return (f"Error: no skill named {name!r}. "
                f"Available skills: {', '.join(sorted(skills))}.")
    _, body = parse_skill(skills[name])
    return body
