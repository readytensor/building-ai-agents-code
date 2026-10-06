"""
Episode 6 — Subagents (skills, carried forward from Ep 5)

Ep 5's skill system — list_skills + load_skill + the skill-provided tools —
carried forward, but adapted to Ep 6's recursive runtime.

In Ep 5 the loaded-skills state was a module-level global (one agent). Ep 6
runs many agents at once, so each run_agent call owns its own `loaded` dict and
its own `tools_by_name`. make_list_skills_tool / make_load_skill_tool bind the
tools to a specific agent's state via closures; the skill-provided tool
implementations (web_search, fetch_url, lint, coverage) are stateless, so they
stay plain module-level functions shared by every worker.

parse_frontmatter is the same tiny `---`-delimited parser Ep 5 used for
SKILL.md; Ep 6 reuses it for the .agents/<name>.md worker configs too (see
agent.py), which is why it lives here as a shared helper.

Imports one-way from tools (`skills → tools`): the @tool decorator, plus
SANDBOX / _safe_path used by the skill-provided tools (lint, coverage) that run
inside the sandbox workspace. The skill files themselves are not under the
sandbox — they live at .skills/ in the episode root, found via Path(".skills").

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

# Skills live at the episode root, alongside agent.py / skills.py — agent
# infrastructure, not part of the toy codebase, so NOT inside initial/ or the
# sandbox. The agent reaches them only via list_skills / load_skill.
_SKILLS_DIR = Path(".skills")


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Parse `---`-delimited YAML frontmatter at the top of a markdown file.
    Returns (frontmatter_dict, body). Tiny hand-rolled parser — no PyYAML
    dependency. Handles scalars, `[a, b]` lists, and `key: |`-style multi-line
    block values. Shared by .skills/<name>/SKILL.md and .agents/<name>.md."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text.strip()
    body = "\n".join(lines[end + 1:]).strip()
    fm: dict = {}
    current_key = None
    for line in lines[1:end]:
        if not line.strip():
            continue
        if line[0] in " \t" and current_key:
            # continuation of a multi-line (`|` / `>`) block value
            fm[current_key] = (fm.get(current_key, "") + "\n" + line.strip()).strip()
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            fm[k] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        elif v in ("|", ">"):
            fm[k] = ""
            current_key = k
        else:
            fm[k] = v
            current_key = k
    return fm, body


def _load_skill_body(name: str) -> dict:
    """Load a skill's parsed metadata + body. Returns {name, description,
    tools (list), body}. Used by load_skill and by run_agent when an
    agent_type pre-loads skills before its first turn."""
    fm, body = parse_frontmatter(_SKILLS_DIR / name / "SKILL.md")
    return {
        "name": fm.get("name", name),
        "description": fm.get("description", ""),
        "tools": fm.get("tools", []),
        "body": body,
    }


def make_list_skills_tool(loaded: dict):
    """Return a list_skills tool that reports which skills THIS agent has
    loaded (loaded is the agent's per-call loaded-skills dict)."""
    @tool(
        "List available skills (name + description for each). Skills are "
        "bundles of procedural knowledge and tools you can load on demand "
        "when their description matches your current task. Cheap — only "
        "metadata is returned, not the skill bodies."
    )
    def list_skills() -> str:
        if not _SKILLS_DIR.exists():
            return "No skills directory at .skills/."
        entries = []
        for skill_dir in sorted(_SKILLS_DIR.iterdir()):
            meta_path = skill_dir / "SKILL.md"
            if not skill_dir.is_dir() or not meta_path.exists():
                continue
            fm, _ = parse_frontmatter(meta_path)
            sname = fm.get("name", skill_dir.name)
            tag = " (LOADED)" if sname in loaded else ""
            entries.append(f"- **{sname}**{tag}: {fm.get('description', '')}")
        return (("Available skills:\n" + "\n".join(entries)) if entries
                else "No skills available.")
    return list_skills


def make_load_skill_tool(loaded: dict, tools_by_name: dict):
    """Return a load_skill tool bound to THIS agent's per-call state: it loads
    a skill's body into `loaded` and registers the skill's tools into
    `tools_by_name` (the agent's live registry), not any shared global."""
    @tool(
        "Load a skill's full body of instructions and register any tools it "
        "provides. Call this when a skill's description matches your task. "
        "The skill's body becomes part of your system prompt; its tools become "
        "available immediately and stay loaded for the rest of the run. "
        "Idempotent — loading twice is a no-op."
    )
    def load_skill(name: str) -> str:
        if name in loaded:
            return f"Skill '{name}' is already loaded."
        if not (_SKILLS_DIR / name / "SKILL.md").exists():
            return (f"Error: skill '{name}' not found. "
                    "Call list_skills() to see available skills.")
        skill = _load_skill_body(name)
        loaded[name] = skill
        new_tools = []
        for tool_name in skill["tools"]:
            if tool_name in _SKILL_TOOLS_REGISTRY:
                tools_by_name[tool_name] = _SKILL_TOOLS_REGISTRY[tool_name]
                new_tools.append(tool_name)
        # Return only a confirmation — NOT the body. The body is injected into
        # the system prompt by system_with_skills() (rebuilt every turn), so
        # returning it here too would put the same text in context twice: once
        # in this tool result (which lingers in message history) and once in
        # the system prompt.
        return (
            f"Skill '{name}' loaded. Tools registered: {new_tools or 'none'}. "
            f"Its instructions are now part of your system prompt."
        )
    return load_skill


def system_with_skills(base_system: str, loaded: dict) -> str:
    """Append each loaded skill's body to the system prompt. The loop calls
    this (wrapped around planning.system_with_plan) each turn. With no skills
    loaded, returns its input unchanged."""
    if not loaded:
        return base_system
    parts = [base_system]
    for name, skill in loaded.items():
        parts.append(f"\n\n[LOADED SKILL: {name}]\n{skill['body']}"
                     f"\n[end skill: {name}]")
    return "".join(parts)


# --- Skill-provided tool implementations (stateless; shared by all workers).
# These only enter a worker's registry once it loads (or pre-loads) the owning
# skill — research brings web_search + fetch_url, verification brings lint +
# coverage.

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
    # User-Agent. Best-effort HTML scrape — fine for a teaching agent.
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
        # DuckDuckGo sometimes wraps a target as //duckduckgo.com/l/?uddg=<url>.
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


# Tools that ONLY register when their owning skill is loaded.
_SKILL_TOOLS_REGISTRY = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "lint": lint,
    "coverage": coverage,
}
