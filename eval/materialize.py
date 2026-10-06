"""Materialize an instance's base state into a fresh git working copy, and
capture the agent's changes as a unified diff.

The git init/commit gives us a clean baseline so `git diff` after the agent runs
is exactly the model_patch. Identity is passed inline so we never depend on the
machine's git config.
"""
import shutil
import subprocess
from pathlib import Path

_IDENTITY = ["-c", "user.email=eval@example.com", "-c", "user.name=eval"]


def _git(repo: Path, *args) -> str:
    return subprocess.run(
        ["git", *_IDENTITY, "-C", str(repo), *args],
        check=True, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    ).stdout


def _rmtree(path: Path) -> None:
    """rmtree that survives Windows read-only files (git objects)."""
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        import os
        import stat
        for p in path.rglob("*"):
            os.chmod(p, stat.S_IWRITE)
        shutil.rmtree(path)


def materialize(base_dir: Path, dest_dir: Path) -> Path:
    """Copy base_dir to dest_dir and commit it as the base state. A leftover
    dest_dir (from a crashed earlier attempt) is replaced, so every attempt
    starts from a pristine copy. Returns dest_dir."""
    if dest_dir.exists():
        _rmtree(dest_dir)
    shutil.copytree(base_dir, dest_dir)
    # If the base already had a .git, drop it so we own the baseline cleanly.
    git_dir = dest_dir / ".git"
    if git_dir.exists():
        _rmtree(git_dir)
    _git(dest_dir, "init", "-q")
    _git(dest_dir, "add", "-A")
    _git(dest_dir, "commit", "-q", "-m", "base")
    return dest_dir


def capture_diff(repo_dir: Path) -> str:
    """Unified diff of all changes since the base commit."""
    _git(repo_dir, "add", "-A")
    return _git(repo_dir, "diff", "--cached")
