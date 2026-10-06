"""Shared types for the eval harness: the Instance an agent is pointed at, the
Verdict scoring produces, and the solve() protocol an agent-under-test satisfies.

Everything downstream (sampling, runner, results) speaks only these types, so
providers differ only in where instance metadata comes from and how scoring runs.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# An agent-under-test: given a working copy and the task text, edit the repo
# in place and return "" (the runner captures the diff itself). The working
# copy is None for container-backed instances, whose workspace is the
# container's own /testbed.
SolveFn = Callable[[Optional[Path], str], str]


@dataclass
class Verdict:
    """The score for one attempt. `passed` uses the real SWE-bench criterion:
    every FAIL_TO_PASS test passes AND every PASS_TO_PASS test still passes."""
    fail_to_pass: dict = field(default_factory=lambda: {"passed": [], "failed": []})
    pass_to_pass: dict = field(default_factory=lambda: {"passed": [], "failed": []})
    details: str = ""

    @property
    def passed(self) -> bool:
        f2p_all_pass = (len(self.fail_to_pass["failed"]) == 0
                        and len(self.fail_to_pass["passed"]) > 0)
        p2p_no_regress = len(self.pass_to_pass["failed"]) == 0
        return f2p_all_pass and p2p_no_regress

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "details": self.details,
        }


@dataclass
class Instance:
    """One problem: a base repo state, the task text, the tests that define
    success, and a scorer that knows how to run them into a Verdict."""
    id: str
    problem_statement: str
    # The base tree to materialize a host working copy from. None means the
    # instance is container-backed: its workspace is the container's own
    # /testbed and nothing exists on the host.
    repo_dir: Optional[Path]
    fail_to_pass: list
    pass_to_pass: list
    scorer: Callable[..., Verdict]
    # Optional hook run before materialization. Providers whose base state is
    # expensive to obtain (a git clone) use this to fetch lazily, so only the
    # instances actually sampled pay the cost. None = repo_dir already exists.
    prepare: Optional[Callable[[], None]] = None
    # Provider-specific attributes used for filtering the pool before sampling
    # (e.g. {"difficulty": "<15 min fix", "repo": "pallets/flask"}).
    meta: dict = field(default_factory=dict)
    # Optional execution-environment hook. Called with the materialized working
    # copy (None for container-backed instances) just before the agent runs;
    # returns a teardown callable. The swebench provider uses this to start the
    # instance's own Docker container and stop it afterwards.
    env_setup: Optional[Callable[[Optional[Path]], Callable[[], None]]] = None
    # Optional diff-capture hook, used by container-backed instances: the diff
    # must be extracted from inside the environment BEFORE teardown. None = the
    # runner diffs the host working copy itself.
    capture: Optional[Callable[[], str]] = None
    # Optional pre-acceptance audit hook: called when the agent requests to
    # stop, returns a list of finding strings (empty = stop accepted). The
    # environment's half of the stop handshake -- providers own the completion
    # contract, the agent loop stays task-agnostic. None = every stop accepted
    # (the classic natural stop).
    audit: Optional[Callable[[], list]] = None

    def verify(self) -> Verdict:
        return self.scorer(self.repo_dir, self.fail_to_pass, self.pass_to_pass)
