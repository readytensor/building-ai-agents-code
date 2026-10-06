"""Run one instance end-to-end: materialize a fresh working copy, hand it to the
agent, score the result, and write the always-keep artifacts.

The agent edits the working copy in place; the diff is captured from the copy's
git state (not the agent's return value), so real and fake agents are handled
uniformly. Scoring runs on that same edited copy. The base repo is never touched
(the agent works on a copy under the batch's own directory), so re-runs are
deterministic.
"""
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

from eval.materialize import materialize, capture_diff
from eval.targets import Instance, SolveFn

# Telemetry files the agent writes to its own cwd (same recording convention as
# the episodes). The runner moves them into the instance's results folder so
# they are preserved per attempt instead of overwritten by the next run.
_TELEMETRY_FILES = ("tool_calls.jsonl", "metrics.json", "final_message.md",
                    "transcript.json", "messages.jsonl")


def run_instance(instance: Instance, solve: SolveFn, batch_dir: Path,
                 run_label: str) -> dict:
    """Materialize, solve, verify, and write verify.json + diff.patch under
    batch_dir/run_label/. Returns a small result dict for the
    summary/scoreboard."""
    inst_dir = batch_dir / run_label
    inst_dir.mkdir(parents=True, exist_ok=True)

    if instance.prepare is not None:
        instance.prepare()  # e.g. fetch an expensive base state (cached)
    if instance.repo_dir is not None:
        work = materialize(instance.repo_dir, inst_dir / "repo")
        # verify() must score the working copy
        scored = replace(instance, repo_dir=work)
    else:
        # Container-backed instance: the workspace is the container's own
        # /testbed; there is nothing to materialize on the host.
        work, scored = None, instance

    # If the provider supplies an execution environment (e.g. the instance's
    # Docker container), stand it up around the agent run and always tear it
    # down -- even when the agent crashes. The diff is captured inside the
    # try: a container-backed instance's changes die with its container.
    teardown = instance.env_setup(work) if instance.env_setup else None
    start = time.monotonic()
    try:
        solve(work, instance.problem_statement, audit=instance.audit)
        diff = instance.capture() if instance.capture else capture_diff(work)
    finally:
        if teardown:
            teardown()
    verdict = scored.verify()
    elapsed = round(time.monotonic() - start, 3)

    (inst_dir / "diff.patch").write_text(diff, encoding="utf-8")
    (inst_dir / "verify.json").write_text(json.dumps(verdict.to_dict(), indent=2),
                                          encoding="utf-8")

    # Collect the agent's telemetry (if it wrote any) into this attempt's folder.
    for name in _TELEMETRY_FILES:
        produced = Path(name)
        if produced.exists():
            shutil.move(str(produced), str(inst_dir / name))

    return {
        "id": instance.id,
        "run_label": run_label,
        "passed": verdict.passed,
        "seconds": elapsed,
        "inst_dir": str(inst_dir),
    }
