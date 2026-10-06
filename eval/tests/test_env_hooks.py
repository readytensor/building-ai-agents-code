import pytest

from eval.runner import run_instance
from eval.scoring import score_pytest
from eval.targets import Instance
from eval.targets import swebench
from eval.tests.conftest import FAIL_TO_PASS, PASS_TO_PASS
from eval.tests.test_swebench_provider import FAKE_RECORD


def _instance(base_repo, env_setup=None):
    return Instance(
        id="calc__add", problem_statement="Fix add.", repo_dir=base_repo,
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        scorer=score_pytest, env_setup=env_setup,
    )


def test_runner_calls_env_setup_and_teardown_around_solve(base_repo, tmp_path,
                                                          solvers):
    events = []

    def setup(work_dir):
        events.append(("setup", work_dir.name))
        return lambda: events.append(("teardown", None))

    run_instance(_instance(base_repo, setup), solvers["fixing"], tmp_path / "b",
                 run_label="calc__add")
    assert [e[0] for e in events] == ["setup", "teardown"]
    assert events[0][1] == "repo"  # setup receives the materialized working copy


def test_teardown_runs_even_when_solve_raises(base_repo, tmp_path):
    events = []

    def setup(work_dir):
        return lambda: events.append("teardown")

    def exploding_solver(repo_dir, task, audit=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_instance(_instance(base_repo, setup), exploding_solver, tmp_path / "b",
                     run_label="x")
    assert events == ["teardown"]


def test_swebench_env_setup_starts_and_stops_container(monkeypatch):
    calls = []
    monkeypatch.setattr(swebench.container, "start",
                        lambda iid: calls.append(("start", iid)) or "cid1")
    monkeypatch.setattr(swebench.container, "stop",
                        lambda cid: calls.append(("stop", cid)))
    inst = swebench.to_instance(FAKE_RECORD)
    teardown = inst.env_setup(None)  # container-backed: no host working copy
    teardown()
    assert calls == [("start", "demo__demo-1"), ("stop", "cid1")]
