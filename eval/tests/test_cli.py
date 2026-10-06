import json

import pytest

from eval import run_eval
from eval.tests.conftest import BASE_FILES


def _md2html_stub(tmp_path):
    base = tmp_path / "md2html"
    base.mkdir()
    for name, content in BASE_FILES.items():
        (base / name).write_text(content, encoding="utf-8")
    return base


def test_cli_fake_agent_end_to_end(tmp_path, monkeypatch):
    base = _md2html_stub(tmp_path)
    specs = [{
        "id": "md2html__demo",
        "problem_statement": "Fix add.",
        "fail_to_pass": ["test_math.py::test_add"],
        "pass_to_pass": ["test_math.py::test_mul"],
    }]
    # Point the local provider at the stub tree + specs.
    from eval.targets.local import build_instances
    monkeypatch.setattr(run_eval, "load_local_instances",
                        lambda: build_instances(base, specs))

    results_root = tmp_path / "results"
    code = run_eval.main([
        "--source", "local", "--agent", "fake-fixing", "--n", "1",
        "--seed", "0", "--results-root", str(results_root), "--timestamp", "t0",
    ])
    assert code == 0

    board = (results_root / "scoreboard.jsonl").read_text().strip().splitlines()
    assert len(board) == 1
    row = json.loads(board[0])
    assert row["pass_at_1"] == 1.0
    assert (results_root / "t0" / "summary.md").exists()
    # The manifest records how the sample was drawn, stratified or flat.
    manifest = json.loads((results_root / "t0" / "manifest.json").read_text())
    assert manifest["stratified"] is False
    verdict = json.loads(
        (results_root / "t0" / "md2html__demo" / "verify.json").read_text())
    assert verdict["passed"] is True


def test_grade_flag_requires_swebench_source():
    with pytest.raises(SystemExit, match="--grade needs --source swebench"):
        run_eval.main(["--source", "local", "--agent", "fake-noop", "--grade"])


def _swebench_run(tmp_path, monkeypatch, extra_args):
    """Run main() with --source swebench but local-style instances and a stub
    grader, so the grade-default behavior is testable token- and Docker-free."""
    base = _md2html_stub(tmp_path)
    specs = [{
        "id": "md2html__demo",
        "problem_statement": "Fix add.",
        "fail_to_pass": ["test_math.py::test_add"],
        "pass_to_pass": ["test_math.py::test_mul"],
    }]
    from eval.targets.local import build_instances
    monkeypatch.setattr(run_eval, "_load_instances",
                        lambda source: build_instances(base, specs))
    graded = []

    def fake_grade(inst_dir, model_label):
        graded.append(inst_dir.name)
        return True

    monkeypatch.setattr(run_eval, "_grade_now", fake_grade)
    results_root = tmp_path / "results"
    code = run_eval.main([
        "--source", "swebench", "--agent", "fake-fixing", "--n", "1",
        "--seed", "0", "--results-root", str(results_root), "--timestamp", "t1",
        *extra_args,
    ])
    assert code == 0
    row = json.loads((results_root / "scoreboard.jsonl").read_text().strip())
    return graded, row


def test_grade_defaults_on_for_swebench(tmp_path, monkeypatch):
    graded, row = _swebench_run(tmp_path, monkeypatch, [])
    assert graded == ["md2html__demo"]
    assert row["grading"] == "official-env-corrected"


def test_no_grade_opts_out_for_swebench(tmp_path, monkeypatch):
    graded, row = _swebench_run(tmp_path, monkeypatch, ["--no-grade"])
    assert graded == []
    assert row["grading"] == "ungraded"


def test_partial_summary_survives_a_fatal_grading_failure(tmp_path, monkeypatch):
    base = _md2html_stub(tmp_path)
    specs = [
        {"id": f"md2html__{name}", "problem_statement": "Fix add.",
         "fail_to_pass": ["test_math.py::test_add"],
         "pass_to_pass": ["test_math.py::test_mul"]}
        for name in ("a", "b")
    ]
    from eval.targets.local import build_instances
    monkeypatch.setattr(run_eval, "_load_instances",
                        lambda source: build_instances(base, specs))

    calls = []

    def failing_grader(inst_dir, model_label):
        calls.append(inst_dir.name)
        if len(calls) > 1:
            raise RuntimeError("grading failed (exit 2): boom")
        return True

    monkeypatch.setattr(run_eval, "_grade_now", failing_grader)
    results_root = tmp_path / "results"
    with pytest.raises(RuntimeError):
        run_eval.main(["--source", "swebench", "--agent", "fake-fixing", "--n", "2",
                       "--seed", "0", "--results-root", str(results_root),
                       "--timestamp", "t2"])
    # The crash left a manifest and a summary covering the completed sample.
    batch = results_root / "t2"
    assert json.loads((batch / "manifest.json").read_text())["n"] == 2
    partial = json.loads((batch / "summary.json").read_text())
    assert partial["aggregate"]["n_instances"] == 1
    assert partial["instances"][0]["passed"] is True


def test_stratified_rejected_with_difficulty_filter():
    with pytest.raises(SystemExit, match="--stratified spans the difficulty buckets"):
        run_eval.main(["--source", "swebench", "--agent", "fake-noop",
                       "--stratified", "--difficulty", "easy"])


def test_stratified_rejected_with_id():
    with pytest.raises(SystemExit, match="--stratified samples a pool"):
        run_eval.main(["--source", "swebench", "--agent", "fake-noop",
                       "--stratified", "--id", "some__instance"])


def test_clean_images_rejected_for_local_source(tmp_path):
    with pytest.raises(SystemExit, match="--clean-images needs --source swebench"):
        run_eval.main([
            "--source", "local", "--agent", "fake-fixing", "--clean-images",
            "--results-root", str(tmp_path / "results"),
        ])
