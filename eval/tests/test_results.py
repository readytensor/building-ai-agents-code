import json

from eval.results import aggregate, write_summary, append_scoreboard, write_manifest


def _results():
    # Two instances, each run twice (repeat=2): one always passes, one never.
    return [
        {"id": "a", "run_label": "a", "passed": True, "seconds": 0.1},
        {"id": "a", "run_label": "a-run2", "passed": True, "seconds": 0.1},
        {"id": "b", "run_label": "b", "passed": False, "seconds": 0.1},
        {"id": "b", "run_label": "b-run2", "passed": False, "seconds": 0.1},
    ]


def test_aggregate_computes_pass_at_1_and_pass_at_k():
    agg = aggregate(_results(), repeat=2)
    assert agg["n_instances"] == 2
    assert agg["pass_at_1"] == 0.5   # a passes on first attempt, b does not
    assert agg["pass_at_k"] == 0.5   # a passes at least once, b never


def test_write_summary_and_manifest(tmp_path):
    agg = aggregate(_results(), repeat=2)
    write_summary(tmp_path, agg, _results())
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()

    manifest = {"agent": "ep5", "model": "gpt-x", "seed": 0,
                "instance_ids": ["a", "b"]}
    write_manifest(tmp_path, manifest)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["agent"] == "ep5"


def test_append_scoreboard_adds_one_row_per_call(tmp_path):
    results_root = tmp_path
    row = {"timestamp": "t1", "agent": "ep5", "pass_at_1": 0.5}
    append_scoreboard(results_root, row)
    append_scoreboard(results_root, {**row, "timestamp": "t2"})
    lines = (results_root / "scoreboard.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert (results_root / "scoreboard.md").exists()
