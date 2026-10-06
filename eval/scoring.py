"""Score a repo against pytest node ids into a Verdict.

Run pytest once over exactly the expected nodes with `-v --tb=no`, which prints
one line per test starting with its exact node id and an outcome word. We read
each expected node's outcome from that output. A node that never appears (a
collection error, a renamed test) counts as failed. Dependency-free: no pytest
plugins, no JSON/JUnit parsing.
"""
import subprocess
from pathlib import Path

from eval.targets import Verdict

# pytest -v prints e.g. "test_math.py::test_add PASSED  [ 50%]". Only these two
# outcomes count as a pass; FAILED / ERROR / SKIPPED / missing all count as fail.
_PASS_WORDS = ("PASSED", "XPASS")


def _outcome(output: str, node_id: str) -> bool:
    """True if `node_id` is reported PASSED in pytest -v output."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(node_id + " ") or line == node_id:
            return any(word in line for word in _PASS_WORDS)
    return False  # node never ran -> treat as failed


def _bucket(output: str, node_ids: list) -> dict:
    passed, failed = [], []
    for node_id in node_ids:
        (passed if _outcome(output, node_id) else failed).append(node_id)
    return {"passed": passed, "failed": failed}


def score_pytest(repo_dir: Path, fail_to_pass: list, pass_to_pass: list) -> Verdict:
    node_ids = list(fail_to_pass) + list(pass_to_pass)
    proc = subprocess.run(
        ["python", "-m", "pytest", *node_ids, "-v", "--tb=no",
         "-p", "no:cacheprovider"],
        cwd=repo_dir, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    output = proc.stdout + proc.stderr
    return Verdict(
        fail_to_pass=_bucket(output, fail_to_pass),
        pass_to_pass=_bucket(output, pass_to_pass),
        # tail is enough to see failures; keeps verify.json small
        details=output[-2000:],
    )
