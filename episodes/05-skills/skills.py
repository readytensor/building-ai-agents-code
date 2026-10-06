"""
Episode 5 — Skills (the new mechanism)

This is what Ep 5 adds to Ep 4. A skill is a lazy-loadable bundle of
procedural knowledge + tools, modeled on Claude Code's skill abstraction:
a `.skills/<name>/SKILL.md` file with YAML frontmatter (name, description,
tools) plus a prose body of instructions.

Two discovery/load tools, always available:

1. list_skills() — walks .skills/, returns each skill's name + description
   (frontmatter only). A cheap discovery surface; bodies are not returned.

2. load_skill(name) — parses the named skill's SKILL.md, records it in
   LOADED_SKILLS so its body rides into the system prompt (see
   system_with_skills), and registers any tools the skill provides into
   LOADED_TOOLS for the rest of the run. Idempotent.

Skill-provided tools only become available once their owning skill loads:
  - web_search / fetch_url  (the `research` skill)
  - lint / coverage         (the `verification` skill)

The system-prompt injection reuses Ep 4's dynamic-system-prompt mechanism:
planning.system_with_plan appends the plan, and system_with_skills here
appends each loaded skill's body. Both live in agent state (not message
history), so they survive compaction and keep the message prefix stable.

Imports one-way from tools (`skills → tools`): the @tool decorator, plus
SANDBOX / _safe_path used by the skill-provided tools (lint, coverage) that run
inside the sandbox workspace. The skill *files* are not under the sandbox —
they live at .skills/ in the episode root, found via Path(".skills"). agent.py
imports from skills, planning, tools, and compaction.

See ../../README.md for context.
"""
import urllib.error
import urllib.parse
import urllib.request
import html as html_module
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

from tools import SANDBOX, _safe_path, tool

# Where skills live: a .skills/ directory at the episode root, alongside
# agent.py / skills.py. Skills are agent infrastructure, not part of the toy
# codebase — so they sit next to the agent's modules, NOT inside initial/ (the
# pristine md2html template) and NOT inside the sandbox the agent edits. A
# consequence: the agent reaches skills only through list_skills / load_skill,
# never through its sandbox-bound file tools (read/grep/bash can't see them).
_SKILLS_DIR = Path(".skills")

# A skill, once loaded, contributes two things for the rest of the run: its
# body (into the system prompt) and its tools (into the live registry). Both
# are module state here — Ep 5 runs a single agent, so a module-level dict is
# enough. (Ep 6 makes these per-call so concurrent workers don't share state.)
# name -> {"name", "description", "tools", "body"}
LOADED_SKILLS: dict[str, dict] = {}
# tool_name -> callable, for skills loaded so far
LOADED_TOOLS: dict[str, callable] = {}


def _parse_skill_md(path: Path) -> dict:
    """Parse a SKILL.md: `---`-delimited YAML frontmatter then a prose body.

    Returns {"name", "description", "tools" (list), "body"}. A tiny hand-rolled
    parser — no PyYAML dependency. Falls back gracefully if the frontmatter is
    missing or malformed (the file's whole text becomes the body)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    meta = {"name": path.parent.name, "description": "", "tools": [],
            "body": text.strip()}
    if not lines or lines[0].strip() != "---":
        return meta
    try:
        end = lines.index("---", 1)
    except ValueError:
        return meta
    meta["body"] = "\n".join(lines[end + 1:]).strip()
    for line in lines[1:end]:
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "tools" and v.startswith("[") and v.endswith("]"):
            meta["tools"] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        elif k in ("name", "description"):
            meta[k] = v
    return meta


@tool(
    "List available skills (name + description for each). Skills are bundles "
    "of procedural knowledge and tools you can load on demand when their "
    "description matches your current task. Call this when starting a task "
    "to see what's available, or whenever you're unsure how to proceed. "
    "Cheap — only metadata is returned, not the skill bodies."
)
def list_skills() -> str:
    if not _SKILLS_DIR.exists():
        return "No skills directory at .skills/."
    entries = []
    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        meta_path = skill_dir / "SKILL.md"
        if not skill_dir.is_dir() or not meta_path.exists():
            continue
        meta = _parse_skill_md(meta_path)
        loaded = " (LOADED)" if meta["name"] in LOADED_SKILLS else ""
        entries.append(f"- **{meta['name']}**{loaded}: {meta['description']}")
    return ("Available skills:\n" + "\n".join(entries) if entries
            else "No skills available.")


@tool(
    "Load a skill's full body of instructions and register any tools it "
    "provides. Call this when a skill's description matches your task. "
    "The skill's body becomes part of your system prompt; its tools become "
    "available immediately and stay loaded for the rest of the run. "
    "Idempotent — loading twice is a no-op."
)
def load_skill(name: str) -> str:
    if name in LOADED_SKILLS:
        return f"Skill '{name}' is already loaded."
    meta_path = _SKILLS_DIR / name / "SKILL.md"
    if not meta_path.exists():
        return (f"Error: skill '{name}' not found. "
                "Call list_skills() to see available skills.")
    skill = _parse_skill_md(meta_path)
    LOADED_SKILLS[name] = skill
    new_tools = []
    for tool_name in skill["tools"]:
        if tool_name in _SKILL_TOOLS_REGISTRY:
            LOADED_TOOLS[tool_name] = _SKILL_TOOLS_REGISTRY[tool_name]
            new_tools.append(tool_name)
    # Return only a confirmation — NOT the body. The body is injected into the
    # system prompt by system_with_skills() (rebuilt every turn), so returning
    # it here too would put the same text in context twice: once in this tool
    # result (which lingers in message history) and once in the system prompt.
    return (
        f"Skill '{name}' loaded. Tools registered: {new_tools or 'none'}. "
        f"Its instructions are now part of your system prompt."
    )


# --- Skill-provided tool implementations. -----------------------------------
# These only enter the live registry once their owning skill is loaded (above).

@tool(
    "Search the web and return the top results as title / url / snippet "
    "blocks. Use this FIRST when you need information that may have changed "
    "since your training, then fetch_url the most authoritative result to "
    "read it in full. Provided by the `research` skill."
)
def web_search(query: str, max_results: int = 5) -> str:
    # Keyless, dependency-free search via DuckDuckGo's HTML endpoint. (Ep 5's
    # original used a provider's server-side search tool; this local version
    # keeps the code portable across LLM providers and runs with no API key.)
    # The endpoint wants a POST with the query as form data and a browser-like
    # User-Agent. It's a best-effort HTML scrape — fine for a teaching agent,
    # not a production search client.
    browser_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    try:
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=urllib.parse.urlencode({"q": query}).encode(),
            headers={"User-Agent": browser_ua},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return f"Error searching for {query!r}: {e.reason}"
    except Exception as e:
        return f"Error searching for {query!r}: {type(e).__name__}: {e}"

    def strip_tags(s: str) -> str:
        return html_module.unescape(re.sub(r"<[^>]+>", "", s)).strip()

    def real_url(href: str) -> str:
        # DuckDuckGo wraps each target as //duckduckgo.com/l/?uddg=<encoded-url>.
        if "uddg=" in href:
            params = parse_qs(urlparse(href).query)
            if params.get("uddg"):
                return unquote(params["uddg"][0])
        return href

    link_re = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
    snippets = snippet_re.findall(body)
    results = []
    for i, m in enumerate(link_re.finditer(body)):
        if i >= max_results:
            break
        title = strip_tags(m.group(2))
        link = real_url(m.group(1))
        snippet = strip_tags(snippets[i]) if i < len(snippets) else ""
        results.append(f"{title}\n  {link}\n  {snippet}".rstrip())
    return "\n\n".join(results) if results else f"No results for {query!r}."


@tool(
    "Fetch the contents of a URL as text. Returns the response body "
    "(decoded as UTF-8, errors replaced). Useful when you have a "
    "specific URL to read (typically after web_search returns one). "
    "Provided by the `research` skill."
)
def fetch_url(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "md2html-agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            body = resp.read()
        text = body.decode("utf-8", errors="replace")
        if len(text) > 50_000:
            return (text[:50_000]
                    + f"\n\n[...truncated; full length was {len(text):,} chars]")
        return text
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} fetching {url}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error fetching {url}: {e.reason}"
    except Exception as e:
        return f"Error fetching {url}: {type(e).__name__}: {e}"


@tool(
    "Run a linter (ruff) over the sandbox. Returns the lint output, or "
    "'clean' if there are no issues. Provided by the `verification` skill."
)
def lint(path: str = ".") -> str:
    p = _safe_path(path)
    result = subprocess.run(  # noqa: S603  # nosec
        ["ruff", "check", str(p)],
        capture_output=True, text=True,
        cwd=SANDBOX, timeout=30,
        encoding="utf-8", errors="replace",
        check=False,
    )
    return (result.stdout + result.stderr).strip() or "clean"


@tool(
    "Run pytest with coverage reporting. Returns the coverage summary. "
    "Useful for verifying new code is covered by tests. Provided by the "
    "`verification` skill."
)
def coverage() -> str:
    result = subprocess.run(  # noqa: S603  # nosec
        ["python", "-m", "pytest", "--cov=md2html", "--cov-report=term-missing",
         "-q"],
        capture_output=True, text=True,
        cwd=SANDBOX, timeout=60,
        encoding="utf-8", errors="replace",
        check=False,
    )
    return (result.stdout + result.stderr).strip() or "(no output)"


# Tools that ONLY register when their owning skill is loaded. load_skill reads
# this to decide which callables to add to LOADED_TOOLS.
_SKILL_TOOLS_REGISTRY: dict[str, callable] = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "lint": lint,
    "coverage": coverage,
}


def system_with_skills(base_system: str) -> str:
    """Append each loaded skill's body to the system prompt. The loop calls
    this (wrapped around planning.system_with_plan) each turn, so loaded-skill
    instructions are always in front of the model. With no skills loaded, this
    returns its input unchanged."""
    if not LOADED_SKILLS:
        return base_system
    parts = [base_system]
    for name, skill in LOADED_SKILLS.items():
        parts.append(f"\n\n[LOADED SKILL: {name}]\n{skill['body']}"
                     f"\n[end skill: {name}]")
    return "".join(parts)
