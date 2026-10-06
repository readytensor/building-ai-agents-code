import json
from pathlib import Path

from eval import official


def _report(iid, f2p_failures, p2p_failures, resolved):
    return {iid: {
        "patch_successfully_applied": True,
        "resolved": resolved,
        "tests_status": {
            "FAIL_TO_PASS": {"success": [], "failure": f2p_failures},
            "PASS_TO_PASS": {"success": [], "failure": p2p_failures},
        },
    }}


def test_parse_report_extracts_verdict_fields(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_report("x", [], ["t/a", "t/b"], False)))
    r = official.parse_report(p, "x")
    assert r == {"resolved_raw": False, "f2p_ok": True,
                 "p2p_failures": ["t/a", "t/b"]}


def test_env_corrected_verdict_against_gold():
    gold = {"f2p_ok": True, "p2p_failures": ["flaky1", "flaky2"]}
    same_as_gold = {"resolved_raw": False, "f2p_ok": True,
                    "p2p_failures": ["flaky1", "flaky2"]}
    subset_of_gold = {"resolved_raw": False, "f2p_ok": True,
                      "p2p_failures": ["flaky1"]}
    real_regression = {"resolved_raw": False, "f2p_ok": True,
                       "p2p_failures": ["flaky1", "broken"]}
    f2p_broken = {"resolved_raw": False, "f2p_ok": False, "p2p_failures": []}
    assert official.env_corrected(same_as_gold, gold) is True
    assert official.env_corrected(subset_of_gold, gold) is True
    assert official.env_corrected(real_regression, gold) is False
    assert official.env_corrected(f2p_broken, gold) is False


def test_write_predictions_uses_first_attempt_only(tmp_path):
    batch = tmp_path / "batch"
    for label, content in [("inst-1", "patch1"), ("inst-1-run2", "patch1b"),
                           ("inst-2", "patch2")]:
        d = batch / label
        d.mkdir(parents=True)
        (d / "diff.patch").write_text(content)
    path = official.write_predictions(batch, "test-model")
    preds = [json.loads(line) for line in path.read_text().splitlines()]
    assert {p["instance_id"]: p["model_patch"] for p in preds} == {
        "inst-1": "patch1", "inst-2": "patch2"}
    assert all(p["model_name_or_path"] == "test-model" for p in preds)


def test_grade_instance_empty_patch_is_unresolved_without_grading(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(official, "GOLD_BASELINE_DIR", tmp_path / "gold_cache")
    inst = tmp_path / "batch" / "inst-1"
    inst.mkdir(parents=True)
    (inst / "diff.patch").write_text("")  # agent finished without changes

    def exploding_runner(*args):
        raise AssertionError("the harness must not run for an empty patch")

    verdict = official.grade_instance(inst, "m", runner=exploding_runner)
    assert verdict["resolved"] is False
    assert "empty patch" in verdict["note"]
    assert json.loads((inst / "official.json").read_text())["resolved"] is False


def test_grade_batch_skips_empty_patches_but_verdicts_them(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "GOLD_BASELINE_DIR", tmp_path / "gold_cache")
    batch = tmp_path / "batch"
    for label, content in [("inst-1", "a real diff"), ("inst-2", "")]:
        d = batch / label
        d.mkdir(parents=True)
        (d / "diff.patch").write_text(content)

    graded = []

    def fake_runner(predictions, run_id, out_dir, ids):
        graded.extend(ids)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for iid in ids:
            (out / f"{iid}.json").write_text(json.dumps(_report(iid, [], [], True)))

    summary = official.grade_batch(batch, "m", runner=fake_runner)
    assert graded.count("inst-2") == 0  # empty patch never reaches the harness
    assert "inst-2" in summary["unresolved"]
    assert json.loads(
        (batch / "inst-2" / "official.json").read_text())["resolved"] is False


def test_grade_instance_grades_one_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "GOLD_BASELINE_DIR", tmp_path / "gold_cache")
    inst = tmp_path / "batch" / "inst-1"
    inst.mkdir(parents=True)
    (inst / "diff.patch").write_text("diff")

    def fake_runner(predictions, run_id, out_dir, ids):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for iid in ids:
            (out / f"{iid}.json").write_text(
                json.dumps(_report(iid, [], ["flaky1"], False)))

    verdict = official.grade_instance(inst, "test-model", runner=fake_runner)
    assert verdict["resolved"] is True  # same failures as gold -> env-corrected pass
    assert json.loads((inst / "official.json").read_text()) == verdict
    pred = json.loads((inst / "predictions.jsonl").read_text())
    assert pred["instance_id"] == "inst-1"
    assert pred["model_name_or_path"] == "test-model"


def test_grade_instance_strips_repeat_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "GOLD_BASELINE_DIR", tmp_path / "gold_cache")
    inst = tmp_path / "batch" / "inst-1-run2"
    inst.mkdir(parents=True)
    (inst / "diff.patch").write_text("diff")

    graded_ids = []

    def fake_runner(predictions, run_id, out_dir, ids):
        graded_ids.extend(ids)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for iid in ids:
            (out / f"{iid}.json").write_text(json.dumps(_report(iid, [], [], False)))

    official.grade_instance(inst, "test-model", runner=fake_runner)
    # The grader must see the real instance id, not the attempt label.
    assert set(graded_ids) == {"inst-1"}
    pred = json.loads((inst / "predictions.jsonl").read_text())
    assert pred["instance_id"] == "inst-1"


def test_grade_batch_end_to_end_with_fake_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "GOLD_BASELINE_DIR", tmp_path / "gold_cache")
    batch = tmp_path / "batch"
    for iid in ("inst-1", "inst-2"):
        d = batch / iid
        d.mkdir(parents=True)
        (d / "diff.patch").write_text("diff")

    calls = []

    def fake_runner(predictions, run_id, out_dir, ids):
        calls.append(predictions)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for iid in ids:
            if predictions == "gold":
                rep = _report(iid, [], ["flaky1"], False)  # env-flaky baseline
            elif iid == "inst-1":
                rep = _report(iid, [], ["flaky1"], False)  # same as gold -> resolved
            else:
                rep = _report(iid, [], ["flaky1", "broken"], False)  # regression
            (out / f"{iid}.json").write_text(json.dumps(rep))

    summary = official.grade_batch(batch, "test-model", runner=fake_runner)
    assert summary["resolved"] == ["inst-1"]
    assert summary["unresolved"] == ["inst-2"]
    assert summary["official_pass_at_1"] == 0.5

    o1 = json.loads((batch / "inst-1" / "official.json").read_text())
    assert o1["resolved"] is True and o1["resolved_raw"] is False
    assert o1["beyond_gold"] == []

    # Gold baselines are cached: a second grade only runs the agent predictions.
    calls.clear()
    official.grade_batch(batch, "test-model", runner=fake_runner)
    assert "gold" not in calls
