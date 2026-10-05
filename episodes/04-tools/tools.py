"""
Episode 2 — Tools

The agent's action space: six general primitives (bash, list_files, read,
write, edit, grep) plus a tiny @tool decorator (~25 lines) that builds each
tool's JSON-schema definition from its Python signature. Every tool resolves
paths inside SANDBOX. `list_files` is a cross-platform alternative to shell
find/ls/dir, so navigation doesn't depend on the host shell.

From Ep 2 onward the tools live here, separate from the agent loop. New
episodes add tools to this file; agent.py imports the registry and rarely
changes. agent.py owns the sandbox *reset* — this file just names the dir.

See ../../README.md for context.
"""
import functools
import inspect
import json
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import get_type_hints

# The working directory every tool is bounded to. agent.py resets it to a clean
# copy of initial/ at the start of each run.
SANDBOX = Path("sandbox")

# --- Tool-call telemetry: record every tool the agent invokes, in order, so we
# can see the path it took and how many calls it made (this varies run to run).
# Recorded in the @tool wrapper below and written to tool_calls.jsonl at the end
# of the run; the harness (run.py) renders the summary.
TOOL_CALLS = []  # list of {"round": n, "tool": name, "args": {...}} in call order
CURRENT_ROUND = 0  # the agent-loop iteration; agent.py sets it each turn so every
# recorded tool call is tagged with the round (model call) it happened in


def write_tool_telemetry():
    """Write the tool calls made this run to tool_calls.jsonl, one JSON object
    per line in call order. Recording only — rendering a summary is left to
    whatever reads the file."""
    with open("tool_calls.jsonl", "w", encoding="utf-8") as f:
        for call in TOOL_CALLS:
            f.write(json.dumps(call) + "\n")


# --- The @tool decorator: function signature -> JSON-schema tool definition.
def tool(description: str):
    """Attach a JSON-schema tool definition to a Python function."""
    json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}

    def decorator(func):
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties, required = {}, []
        for name, param in sig.parameters.items():
            t = hints.get(name, str)
            properties[name] = {"type": json_types.get(t, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Record the call before running it, so the path stays correct even
            # if the tool raises.
            bound = sig.bind(*args, **kwargs)
            TOOL_CALLS.append({"round": CURRENT_ROUND, "tool": func.__name__, "args": dict(bound.arguments)})
            return func(*args, **kwargs)

        wrapper.tool_definition = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        return wrapper
    return decorator


# --- The working tools. All paths resolve inside SANDBOX.
def _safe_path(path: str) -> Path:
    """Resolve `path` inside SANDBOX. Raises if it escapes."""
    resolved = (SANDBOX / path).resolve()
    resolved.relative_to(SANDBOX.resolve())  # raises ValueError if outside
    return resolved


@tool("Execute a shell command in the working directory and return its output.")
def bash(command: str) -> str:
    # Failures (non-zero exit) come back as tool output so the model can read
    # the error and adapt; crashing the agent on non-zero exit would defeat that.
    #
    # cwd only sets the STARTING directory -- it is not a security boundary.
    # shell=True gives the model a real shell that can cd anywhere and touch
    # anything this process can. True isolation needs a container or an OS
    # sandbox (the way real agents such as Claude Code do it); out of scope for
    # this toy, so run it on code you trust or inside a throwaway VM/container.
    proc = subprocess.Popen(  # noqa: S602  # nosec
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=SANDBOX, encoding="utf-8", errors="replace",
        start_new_session=(os.name != "nt"),  # POSIX: own group so we can kill the whole tree
    )
    try:
        output = proc.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)  # boilerplate: shell=True orphans grandchildren
        return (
            "Error: command timed out after 30s and was killed (whole process "
            "tree). Avoid long-running or interactive commands, watch for code "
            "that can loop forever, and scope file searches to the working directory."
        )
    output = (output or "").strip()
    if len(output) > 20_000:                 # cap transcript growth from chatty commands
        output = output[:20_000] + "\n...[truncated]"
    if proc.returncode:                      # surface failures so the model can adapt
        output += f"\n(exit code {proc.returncode})"
    return output or "(no output)"


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a timed-out shell AND its descendants (cross-platform boilerplate).

    With shell=True the real command runs as a child of the shell; killing
    only the shell can leave that child alive, still holding the output pipe
    -- which deadlocks the drain (an unbounded loop would then hang forever).
    taskkill /T on Windows and killpg on POSIX take descendants down too.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    if proc.poll() is None:   # if the tree-kill missed, at least kill the shell
        proc.kill()

    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass


# Directories that are never task content: VCS internals, caches, build output.
# list_files and grep both skip them, so their output caps (200 files, 50
# matches) are spent on real source instead of noise -- on a big repo a
# vendored build/ tree alone can eat the whole budget.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", ".ruff_cache",
             "build", "dist", "node_modules", ".tox", ".eggs"}


@tool("List files under a path (recursive), one relative path per line — a reliable cross-platform alternative to shell find/ls/dir. Skips caches and VCS dirs.")
def list_files(path: str = ".") -> str:
    sandbox = SANDBOX.resolve()
    root = _safe_path(path)
    if root.is_file():
        return str(root.relative_to(sandbox))
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            files.append(str(p.relative_to(sandbox)))
            if len(files) >= 200:
                files.append("... (truncated at 200 files)")
                break
    return "\n".join(files) if files else "(no files)"


@tool("Read a file's contents, prefixed with line numbers. For large files, pass offset (1-based line to start at) and limit (max lines) to read just a slice instead of the whole file.")
def read(path: str, offset: int = 1, limit: int = 0) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if p.is_dir():
        return f"Error: {path} is a directory. Use bash to list its contents."
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    # Number the whole file before slicing, so a slice keeps its real line
    # numbers (line 200 is still labeled 200) and matches grep/traceback output.
    numbered = [f"{i+1:5d}\t{line}" for i, line in enumerate(lines)]
    start = max(offset, 1) - 1
    end = start + limit if limit > 0 else len(numbered)
    selected = numbered[start:end]
    if not selected:
        return f"Error: {path} has {len(lines)} lines; offset {offset} is past the end."
    if len(selected) < len(lines):
        selected.append(f"(showing lines {start + 1}-{start + len(selected)} of {len(lines)})")
    return "\n".join(selected)


@tool("Write content to a file, overwriting any existing content. Creates parent directories.")
def write(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}."


@tool("Replace old_string with new_string in a file. Replaces a single occurrence and errors if old_string isn't unique; pass replace_all=true to replace every occurrence (e.g. renaming a symbol).")
def edit(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if p.is_dir():
        return f"Error: {path} is a directory."
    text = p.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}."
    if count > 1 and not replace_all:
        return f"Error: old_string appears {count} times in {path}; pass replace_all=true to replace all, or add more context to make it unique."
    p.write_text(text.replace(old_string, new_string), encoding="utf-8")
    return f"Replaced {count} occurrence(s) in {path}."


@tool("Search for a regex pattern under a path. Returns matches as relative/path:line: text. Skips caches and VCS dirs.")
def grep(pattern: str, path: str = ".") -> str:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    sandbox = SANDBOX.resolve()
    root = _safe_path(path)
    if root.is_file():
        files = [root]
    else:
        files = [p for p in root.rglob("*")
                 if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)]
    results = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = f.relative_to(sandbox)
                    results.append(f"{rel}:{i}: {line[:200]}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n... (truncated at 50 matches)"
        except Exception:
            continue  # skip binary / unreadable
    return "\n".join(results) if results else f"No matches for {pattern!r}."


# --- Tool registry: name -> callable, plus the list of schemas for the LLM.
TOOLS = [bash, list_files, read, write, edit, grep]
TOOLS_BY_NAME = {t.__name__: t for t in TOOLS}
TOOL_DEFS = [t.tool_definition for t in TOOLS]
