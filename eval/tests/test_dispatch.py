"""The dispatcher runs one run_eval subprocess per instance, at most K at a
time. These tests drive it with a fake spawner (the codebase's injectable-
runner convention), so scheduling, resume, consolidation, and failure handling
are all covered token- and subprocess-free.
"""
import json
from pathlib import Path

from eval import dispatch


class FakeProc:
    """Stands in for subprocess.Popen: 'runs' for a fixed number of polls,
    then finishes by writing the artifacts a real run_eval would leave."""

    def __init__(self, iid, batch_dir, polls_left=2, exit_code=0, passed=True):
        self.iid = iid
        self.batch_dir = Path(batch_dir)
        self.polls_left = polls_left
        self.exit_code = exit_code
        self.passed = passed

    def poll(self):
        if self.polls_left > 0:
            self.polls_left -= 1
            return None
        if self.exit_code == 0:
            _write_run_artifacts(self.batch_dir, self.iid, self.passed)
        return self.exit_code


def _write_run_artifacts(batch_dir, iid, passed=True, seconds=1.0):
    """What a finished run_eval leaves behind: its own sub-batch dir with a
    summary.json whose single instance row carries the score of record."""
    sub = Path(batch_dir) / iid
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "manifest.json").write_text(json.dumps({
        "agent": "ep5", "model": "test-model", "agent_git_sha": "abc1234",
    }), encoding="utf-8")
    (sub / "summary.json").write_text(json.dumps({
        "aggregate": {"n_instances": 1, "repeat": 1,
                      "pass_at_1": 1.0 if passed else 0.0,
                      "pass_at_k": 1.0 if passed else 0.0,
                      "mean_seconds": seconds},
        "instances": [{"id": iid, "run_label": iid, "passed": passed,
                       "seconds": seconds, "inst_dir": str(sub / iid)}],
    }), encoding="utf-8")


def _run(tmp_path, ids, workers=2, procs=None, **kwargs):
    """Drive dispatch.run_dispatch with a fake spawner; returns (result,
    spawn_calls, max_concurrent)."""
    results_root = tmp_path / "results"
    batch_dir = results_root / "batch"
    running, spawn_calls, peak = [], [], [0]

    def spawn(iid, cmd, cwd, env, log_path):
        spawn_calls.append({"iid": iid, "cmd": cmd, "cwd": cwd, "env": env})
        proc = (procs or {}).get(iid) or FakeProc(iid, batch_dir)
        running.append(proc)
        live = [p for p in running if p.polls_left > 0]
        peak[0] = max(peak[0], len(live))
        return proc

    result = dispatch.run_dispatch(
        ids, batch_dir=batch_dir, results_root=results_root,
        source="swebench", agent="ep5", keep="all", workers=workers,
        spawn=spawn, sleep=lambda s: None, **kwargs)
    return result, spawn_calls, peak[0]


def test_runs_every_id_with_bounded_concurrency(tmp_path):
    ids = [f"repo__inst-{n}" for n in range(5)]
    result, spawn_calls, peak = _run(tmp_path, ids, workers=2)
    assert sorted(c["iid"] for c in spawn_calls) == sorted(ids)
    assert peak <= 2
    assert result["failed"] == []
    assert len(result["results"]) == 5


def test_worker_gets_own_cwd_and_repo_on_pythonpath(tmp_path):
    ids = ["repo__inst-0", "repo__inst-1"]
    _, spawn_calls, _ = _run(tmp_path, ids)
    cwds = {c["cwd"] for c in spawn_calls}
    assert len(cwds) == 2  # one private cwd per worker, or telemetry collides
    repo_root = str(Path(dispatch.__file__).resolve().parents[1])
    for c in spawn_calls:
        assert repo_root in (c["env"]["PYTHONPATH"].split(";")
                             + c["env"]["PYTHONPATH"].split(":"))
        assert "--id" in c["cmd"] and c["iid"] in c["cmd"]
        assert "--keep" in c["cmd"] and "all" in c["cmd"]


def test_skips_instances_already_complete(tmp_path):
    ids = ["repo__done", "repo__todo"]
    batch_dir = tmp_path / "results" / "batch"
    _write_run_artifacts(batch_dir, "repo__done")
    result, spawn_calls, _ = _run(tmp_path, ids)
    assert [c["iid"] for c in spawn_calls] == ["repo__todo"]
    # The completed instance still counts in the consolidated results.
    assert sorted(r["id"] for r in result["results"]) == sorted(ids)


def test_consolidates_one_summary_and_one_scoreboard_row(tmp_path):
    ids = ["repo__a", "repo__b", "repo__c"]
    procs = {"repo__b": FakeProc("repo__b", tmp_path / "results" / "batch",
                                 passed=False)}
    result, _, _ = _run(tmp_path, ids, procs=procs)

    batch_dir = tmp_path / "results" / "batch"
    summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["n_instances"] == 3
    assert summary["aggregate"]["pass_at_1"] == round(2 / 3, 4)

    board = ((tmp_path / "results" / "scoreboard.jsonl").read_text(encoding="utf-8")
             .strip().splitlines())
    assert len(board) == 1
    row = json.loads(board[0])
    assert row["n"] == 3
    assert row["model"] == "test-model"
    assert row["grading"] == "official-env-corrected"
    assert result["failed"] == []


def test_worker_failure_is_reported_not_scored(tmp_path):
    ids = ["repo__ok", "repo__boom"]
    procs = {"repo__boom": FakeProc("repo__boom", tmp_path / "results" / "batch",
                                    exit_code=3)}
    result, _, _ = _run(tmp_path, ids, procs=procs)

    assert result["failed"] == ["repo__boom"]
    # The failed worker is excluded from the aggregate (an infra failure is
    # not an agent failure) and blocks the scoreboard row: rerunning the
    # dispatcher resumes the missing instance and appends the row then.
    summary = json.loads((tmp_path / "results" / "batch" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["aggregate"]["n_instances"] == 1
    assert not (tmp_path / "results" / "scoreboard.jsonl").exists()
    assert result["exit_code"] == 1


def test_disk_floor_stops_new_workers_and_blocks_scoreboard(tmp_path):
    # Below the floor nothing spawns: a full disk kills the WSL VM under
    # Docker (the 2026-07-03 crash), so refusing the worker is the safe move.
    ids = ["repo__a", "repo__b"]
    result, spawn_calls, _ = _run(tmp_path, ids, free_gb=lambda p: 5.0)
    assert spawn_calls == []
    assert result["skipped"] == ids
    assert result["exit_code"] == 1
    assert not (tmp_path / "results" / "scoreboard.jsonl").exists()


def test_disk_floor_lets_running_workers_finish(tmp_path):
    # Disk drops under the floor after the first spawn: the in-flight worker
    # runs to completion (its result counts); the rest are skipped unrun.
    ids = ["repo__a", "repo__b", "repo__c"]
    reads = iter([100.0, 5.0, 5.0, 5.0])
    result, spawn_calls, _ = _run(tmp_path, ids, workers=1,
                                  free_gb=lambda p: next(reads))
    assert [c["iid"] for c in spawn_calls] == ["repo__a"]
    assert [r["id"] for r in result["results"]] == ["repo__a"]
    assert result["skipped"] == ["repo__b", "repo__c"]
    assert result["exit_code"] == 1
