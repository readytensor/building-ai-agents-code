"""File operations executed INSIDE a SWE-bench instance container.

The harness never imports this module at agent runtime. eval/container.py
sends its source over `docker exec -i <cid> python -` (stdin), with one
trailing `dispatch(...)` call appended, so each operation runs inside the
container against its own /testbed checkout -- the canonical scaffold
pattern: the workspace lives in the container, and only text crosses the
boundary.

Semantics mirror the episodes' host-side file tools (episodes/05-skills/
tools.py) exactly -- same outputs, same error strings -- so the model sees no
difference between a local host run and a container run.

It is also a plain importable module, which is how the offline tests exercise
it: dispatch(payload, root=some_tmp_dir).
"""
import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/testbed")

# Same skip set as the episodes' tools.py: never task content, and on big
# repos this junk would otherwise eat the 200-file / 50-match output caps.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", ".ruff_cache",
             "build", "dist", "node_modules", ".tox", ".eggs"}


def _safe_path(root: Path, path: str) -> Path:
    """Resolve `path` inside root. The agent sees container paths, so both
    "/testbed/x/y.py" (copied from tracebacks and shell output) and plain
    relative paths are accepted. Raises if the result escapes root."""
    if path == "/testbed" or path == str(root):
        path = "."
    elif path.startswith("/testbed/"):
        path = path[len("/testbed/"):]
    resolved = (root / path).resolve()
    resolved.relative_to(root.resolve())  # raises ValueError if outside
    return resolved


def list_files(root: Path, path: str = ".") -> str:
    top = _safe_path(root, path)
    if top.is_file():
        return str(top.relative_to(root.resolve()))
    files = []
    for p in sorted(top.rglob("*")):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            files.append(str(p.relative_to(root.resolve())))
            if len(files) >= 200:
                files.append("... (truncated at 200 files)")
                break
    return "\n".join(files) if files else "(no files)"


def read(root: Path, path: str, offset: int = 1, limit: int = 0) -> str:
    p = _safe_path(root, path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if p.is_dir():
        return f"Error: {path} is a directory. Use bash to list its contents."
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    numbered = [f"{i+1:5d}\t{line}" for i, line in enumerate(lines)]
    start = max(offset, 1) - 1
    end = start + limit if limit > 0 else len(numbered)
    selected = numbered[start:end]
    if not selected:
        return (f"Error: {path} has {len(lines)} lines; "
                f"offset {offset} is past the end.")
    if len(selected) < len(lines):
        selected.append(f"(showing lines {start + 1}-{start + len(selected)} "
                        f"of {len(lines)})")
    return "\n".join(selected)


def write(root: Path, path: str, content: str) -> str:
    p = _safe_path(root, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}."


def edit(root: Path, path: str, old_string: str, new_string: str,
         replace_all: bool = False) -> str:
    p = _safe_path(root, path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if p.is_dir():
        return f"Error: {path} is a directory."
    text = p.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path}."
    if count > 1 and not replace_all:
        return (f"Error: old_string appears {count} times in {path}; "
                "pass replace_all=true to replace all, "
                "or add more context to make it unique.")
    p.write_text(text.replace(old_string, new_string), encoding="utf-8")
    return f"Replaced {count} occurrence(s) in {path}."


def grep(root: Path, pattern: str, path: str = ".") -> str:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    top = _safe_path(root, path)
    if top.is_file():
        files = [top]
    else:
        files = [p for p in top.rglob("*")
                 if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)]
    results = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = f.relative_to(root.resolve())
                    results.append(f"{rel}:{i}: {line[:200]}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n... (truncated at 50 matches)"
        except Exception:
            continue  # skip binary / unreadable
    return "\n".join(results) if results else f"No matches for {pattern!r}."


# ---------------------------------------------------------------------------
# repo_map: a generated orientation map of the repository (stdlib ast only).
#
# Level 1 (path "." or omitted) answers "what is this project?": the README
# and runner-config files that exist, the package tree with module counts and
# __init__ docstring one-liners, and where the tests live. It reports
# EVIDENCE, not conclusions -- choosing the test command is the model's job.
# Level 2 (path=<subdir>) answers "what is in here?": per-module class and
# function signatures, for orienting inside a package before editing it.
# ---------------------------------------------------------------------------

_DIR_CAP = 150       # level-1 package-tree lines before truncation
_TEST_CAP = 30       # level-1 test-location lines
_TOPFILE_CAP = 30    # level-1 top-level file names
_FILE_CAP = 40       # level-2 modules per call
_SIG_CAP = 400       # level-2 signature lines
_MAP_CHAR_CAP = 8_000  # hard cap on any map: the root map rides in EVERY
                       # turn's system prompt, so size discipline matters
_RUNNER_FILES = ("tox.ini", "pytest.ini", "setup.cfg", "pyproject.toml",
                 "noxfile.py", "Makefile", "runtests.py")


def _parse(path: Path):
    """ast.parse a source file, or None if it doesn't parse (a repo can
    legitimately contain broken or version-specific files)."""
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None


def _first_doc_line(tree) -> str:
    doc = (ast.get_docstring(tree) or "").strip()
    return doc.splitlines()[0][:100] if doc else ""


def _sig(node) -> str:
    """A compact one-line signature: argument names only (no annotations or
    defaults -- orientation, not documentation)."""
    args = [a.arg for a in node.args.posonlyargs + node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    args += [a.arg for a in node.args.kwonlyargs]
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}def {node.name}({', '.join(args)})"


def _walk_pruned(top: Path):
    """os.walk with junk and hidden directories pruned DURING the walk (mutating
    dirnames in place stops the descent) -- rglob-then-filter would still crawl
    all of .git or node_modules on a big repo. Sorted for stable output."""
    for cur, dirnames, filenames in os.walk(top):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        yield Path(cur), sorted(filenames)


def _mapped_dirs(rootr: Path) -> list:
    """(directory, direct .py module count) for every directory under root
    that directly holds .py files."""
    return [(d, sum(1 for f in files if f.endswith(".py")))
            for d, files in _walk_pruned(rootr)
            if d != rootr and any(f.endswith(".py") for f in files)]


def _dir_doc(d: Path) -> str:
    """The __init__.py docstring one-liner for a directory, or ""."""
    init = d / "__init__.py"
    if not init.exists():
        return ""
    tree = _parse(init)
    return _first_doc_line(tree) if tree else ""


def _tree_lines(rootr: Path, entries: list, depth: int = None) -> list:
    """Render the package tree, optionally collapsed to a directory depth.

    Uncollapsed (depth None): one line per directory. Collapsed: every entry
    is folded into its first `depth` path components, and each surviving
    prefix line aggregates the module count of everything beneath it -- so a
    huge tree shows ALL its top branches shallowly instead of an alphabetical
    prefix deeply (django: conf/locale/* was crowding out db/ and forms/)."""
    if depth is None:
        lines = []
        for d, n in entries:
            rel = str(d.relative_to(rootr)).replace("\\", "/")
            doc = _dir_doc(d)
            lines.append(f"  {rel}/ ({n} modules)" + (f" -- {doc}" if doc else ""))
        return lines
    groups = {}  # prefix parts -> [modules, dirs]
    for d, n in entries:
        prefix = d.relative_to(rootr).parts[:depth]
        stat = groups.setdefault(prefix, [0, 0])
        stat[0] += n
        stat[1] += 1
    lines = []
    for prefix in sorted(groups):
        modules, ndirs = groups[prefix]
        rel = "/".join(prefix)
        count = (f"({modules} modules in {ndirs} dirs)" if ndirs > 1
                 else f"({modules} modules)")
        doc = _dir_doc(rootr.joinpath(*prefix))
        lines.append(f"  {rel}/ {count}" + (f" -- {doc}" if doc else ""))
    return lines


def repo_map(root: Path, path: str = ".") -> str:
    top = _safe_path(root, path)
    if not top.is_dir():
        return f"Error: {path} is not a directory."
    rootr = root.resolve()
    out = _map_overview(rootr) if top == rootr else _map_subtree(rootr, top)
    if len(out) > _MAP_CHAR_CAP:
        out = out[:_MAP_CHAR_CAP] + ("\n... (map truncated; call "
                                     "repo_map(path=...) on a subtree)")
    return out


def _map_overview(rootr: Path) -> str:
    lines = []
    readmes = sorted(p.name for p in rootr.glob("README*") if p.is_file())
    lines.append("README: " + (", ".join(readmes) if readmes else "(none found)"))
    top_files = sorted(p.name for p in rootr.iterdir() if p.is_file())
    shown = ", ".join(top_files[:_TOPFILE_CAP])
    if len(top_files) > _TOPFILE_CAP:
        shown += f", ... (+{len(top_files) - _TOPFILE_CAP} more)"
    lines.append("Top-level files: " + (shown if top_files else "(none)"))
    runner = [n for n in _RUNNER_FILES if (rootr / n).exists()]
    runner += [str(p.relative_to(rootr)).replace("\\", "/")
               for p in sorted(rootr.glob("*/runtests.py"))]
    if runner:
        lines.append("Test/build config found: " + ", ".join(runner))
    lines.append("")
    lines.append("Package tree (directories with Python modules; "
                 "one-liners from __init__.py):")
    entries = _mapped_dirs(rootr)
    test_dirs = [str(d.relative_to(rootr)).replace("\\", "/") for d, _ in entries
                 if d.name.startswith("test") or (d / "conftest.py").exists()]
    # Depth-adaptive collapse: a flat listing of a huge tree burns the cap on
    # alphabetical noise and truncates the real packages out entirely.
    tree = _tree_lines(rootr, entries)
    for depth in (2, 1):
        if len(tree) <= _DIR_CAP:
            break
        tree = _tree_lines(rootr, entries, depth)
    lines += tree[:_DIR_CAP]
    if len(tree) > _DIR_CAP:
        lines.append(f"  ... ({len(tree) - _DIR_CAP} more directories; "
                     "call repo_map(path=...) to inspect one)")
    lines.append("")
    lines.append("Test locations (test* directories or a conftest.py):")
    if test_dirs:
        lines += [f"  {t}/" for t in test_dirs[:_TEST_CAP]]
        if len(test_dirs) > _TEST_CAP:
            lines.append(f"  ... ({len(test_dirs) - _TEST_CAP} more)")
    else:
        lines.append("  (none found)")
    return "\n".join(lines)


def _map_subtree(rootr: Path, top: Path) -> str:
    files = [d / f for d, names in _walk_pruned(top)
             for f in names if f.endswith(".py")]
    if not files:
        return f"No Python modules under {str(top.relative_to(rootr))}."
    lines = []
    for f in files[:_FILE_CAP]:
        rel = str(f.relative_to(rootr)).replace("\\", "/")
        tree = _parse(f)
        if tree is None:
            lines.append(f"{rel} (could not parse)")
            continue
        doc = _first_doc_line(tree)
        lines.append(rel + (f" -- {doc}" if doc else ""))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append(f"  {_sig(node)}")
            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(b.id for b in node.bases if isinstance(b, ast.Name))
                lines.append(f"  class {node.name}({bases})" if bases
                             else f"  class {node.name}")
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        lines.append(f"    {_sig(sub)}")
        if len(lines) >= _SIG_CAP:
            lines.append(f"... (truncated at {_SIG_CAP} lines; "
                         "call repo_map on a deeper path)")
            break
    if len(files) > _FILE_CAP:
        lines.append(f"... ({len(files) - _FILE_CAP} more modules not shown)")
    return "\n".join(lines)


_OPS = {"list_files": list_files, "read": read, "write": write, "edit": edit,
        "grep": grep, "repo_map": repo_map}


def dispatch(payload: dict, root: Path = ROOT) -> str:
    """Run one operation described by {"op": name, **kwargs}. Errors come back
    as strings -- the same convention the agent loop uses for tool failures."""
    payload = dict(payload)
    op = payload.pop("op")
    try:
        return _OPS[op](root, **payload)
    except Exception as e:
        return f"Error executing {op}: {type(e).__name__}: {e}"


if __name__ == "__main__" and len(sys.argv) > 1:  # direct CLI use; the piped
    print(dispatch(json.loads(sys.argv[1])), end="")  # form appends its own call
