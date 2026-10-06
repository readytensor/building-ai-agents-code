import json

from eval.runner import run_instance
from eval.scoring import score_pytest
from eval.targets import Instance, Verdict
from eval.tests.conftest import FAIL_TO_PASS, PASS_TO_PASS


def _make_instance(base_repo):
    return Instance(
        id="calc__add",
        problem_statement="Fix add so test_add passes.",
        repo_dir=base_repo,
        fail_to_pass=FAIL_TO_PASS,
        pass_to_pass=PASS_TO_PASS,
        scorer=score_pytest,
    )


def test_fixing_solver_yields_a_passing_result(base_repo, tmp_path, solvers):
    inst = _make_instance(base_repo)
    out = tmp_path / "batch"
    result = run_instance(inst, solvers["fixing"], out, run_label="calc__add")

    assert result["passed"] is True
    verdict = json.loads((out / "calc__add" / "verify.json").read_text())
    assert verdict["passed"] is True
    assert "return a + b" in (out / "calc__add" / "diff.patch").read_text()


def test_noop_solver_yields_a_failing_result(base_repo, tmp_path, solvers):
    inst = _make_instance(base_repo)
    out = tmp_path / "batch"
    result = run_instance(inst, solvers["noop"], out, run_label="calc__add")

    assert result["passed"] is False
    assert (out / "calc__add" / "diff.patch").read_text() == ""


def test_runner_collects_agent_telemetry_files(base_repo, tmp_path, solvers,
                                               monkeypatch):
    monkeypatch.chdir(tmp_path)  # telemetry lands in cwd; keep the test isolated

    def telemetry_solver(repo_dir, task, audit=None):
        (tmp_path / "tool_calls.jsonl").write_text('{"tool": "bash"}\n')
        (tmp_path / "metrics.json").write_text('{"agents": []}')
        (tmp_path / "final_message.md").write_text("all done")
        (tmp_path / "transcript.json").write_text('[{"role": "system"}]')
        return ""

    inst = _make_instance(base_repo)
    run_instance(inst, telemetry_solver, tmp_path / "batch", run_label="calc__add")
    assert (tmp_path / "batch" / "calc__add" / "tool_calls.jsonl").exists()
    assert (tmp_path / "batch" / "calc__add" / "metrics.json").exists()
    assert (tmp_path / "batch" / "calc__add" / "final_message.md").exists()
    assert (tmp_path / "batch" / "calc__add" / "transcript.json").exists()
    assert not (tmp_path / "tool_calls.jsonl").exists()  # moved, not copied


def test_runner_does_not_mutate_the_base_repo(base_repo, tmp_path, solvers):
    inst = _make_instance(base_repo)
    run_instance(inst, solvers["fixing"], tmp_path / "batch", run_label="calc__add")
    # The base repo's calc.py must remain broken; the agent edits a working copy.
    assert "return a - b" in (base_repo / "calc.py").read_text()


def test_container_backed_instance_has_no_host_copy_and_captures_before_teardown(
        tmp_path):
    events = []
    inst = Instance(
        id="demo__demo-1", problem_statement="fix it", repo_dir=None,
        fail_to_pass=["t::a"], pass_to_pass=[],
        scorer=lambda repo_dir, f2p, p2p: Verdict(details="ungraded"),
        env_setup=lambda work: (events.append(("setup", work)),
                                lambda: events.append(("teardown",)))[1],
        capture=lambda: events.append(("capture",)) or "the diff",
    )

    def solver(repo_dir, task, audit=None):
        assert repo_dir is None  # nothing on the host to point the agent at
        return ""

    run_instance(inst, solver, tmp_path / "batch", run_label="demo__demo-1")
    inst_dir = tmp_path / "batch" / "demo__demo-1"
    assert not (inst_dir / "repo").exists()  # no materialized working copy
    assert (inst_dir / "diff.patch").read_text() == "the diff"
    # The diff must be extracted while the container is still alive.
    assert events == [("setup", None), ("capture",), ("teardown",)]
