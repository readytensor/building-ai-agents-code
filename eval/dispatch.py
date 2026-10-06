"""Run a batch of eval instances in parallel: one run_eval subprocess per
instance, at most K at a time.

    python -m eval.dispatch --batch rebaseline-v3 --workers 3 --ids id1 id2 ...

Parallelism is process-level by necessity, not preference: the agent writes
its telemetry (tool_calls.jsonl, metrics.json, final_message.md) to its own
cwd, and the episode modules it reuses hold per-run global state (TOOL_CALLS,
LOADED_SKILLS, container.ACTIVE). Threads or a shared cwd would cross-
contaminate samples; a subprocess with a private cwd contains each one.

Each worker is a full, unmodified run_eval invocation (solve -> grade with the
usual --grade default), writing into its own sub-batch dir under the batch
dir. The dispatcher only schedules, resumes, and consolidates:

  <batch_dir>/<instance_id>/            one worker's run_eval batch dir
  <batch_dir>/<instance_id>.log         that worker's live output (tail -f it)
  <batch_dir>/_cwd/<instance_id>/       that worker's private cwd
  <batch_dir>/summary.json|.md          consolidated across all instances

Resume: an instance whose sub-batch dir already has a summary.json is
complete (run_eval writes it after grading) and is not re-run, so a killed
batch picks up where it left off. A failed worker is an infra failure, not an
agent failure: it is excluded from the aggregate and blocks the consolidated
scoreboard row until a re-run completes the batch.

Disk floor: no new worker starts with less than MIN_FREE_GB free. A full disk
does not fail politely -- it killed the WSL VM under Docker mid-batch on
2026-07-03 and every in-flight worker hung forever. Skipped instances are
left unrun (resume after freeing space); like failures, they block the
scoreboard row.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from eval.results import aggregate, append_scoreboard, write_summary

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Enough headroom for K in-flight instance images (~1-3GB each) plus one
# pull's transient (compressed + extracted layers coexist briefly).
MIN_FREE_GB = 20


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _worker_cmd(iid: str, batch_dir: Path, source: str, agent: str,
                keep: str) -> list:
    return [sys.executable, "-m", "eval.run_eval",
            "--source", source, "--agent", agent, "--id", iid, "--n", "1",
            "--keep", keep, "--results-root", str(batch_dir.resolve()),
            "--timestamp", iid]


def _spawn(iid: str, cmd: list, cwd: Path, env: dict, log_path: Path):
    """Start one worker with its output teed to a per-instance log file. The
    parent's file handle is closed right away; the child keeps its own."""
    with open(log_path, "a", encoding="utf-8") as log:
        return subprocess.Popen(cmd, cwd=str(cwd), env=env,
                                stdout=log, stderr=subprocess.STDOUT)


def _is_complete(batch_dir: Path, iid: str) -> bool:
    """run_eval writes its summary.json only after the sample is solved AND
    graded, so its presence is the one honest completion marker."""
    return (batch_dir / iid / "summary.json").exists()


def _collect_result(batch_dir: Path, iid: str) -> dict:
    summary = json.loads(
        (batch_dir / iid / "summary.json").read_text(encoding="utf-8"))
    return summary["instances"][0]


def run_dispatch(ids: list, *, batch_dir: Path, results_root: Path,
                 source: str = "swebench", agent: str = "ep5", keep: str = "all",
                 workers: int = 3, spawn=_spawn, sleep=time.sleep,
                 free_gb=_free_gb) -> dict:
    """Run every id to completion (skipping ones already complete), at most
    `workers` at a time. Returns {results, failed, skipped, exit_code}."""
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    done = [i for i in ids if _is_complete(batch_dir, i)]
    queue = [i for i in ids if i not in done]
    for iid in done:
        print(f"[dispatch] {iid}: already complete, skipping", flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    running, failed, skipped = {}, [], []  # running: iid -> Popen-like
    while queue or running:
        while queue and len(running) < workers:
            free = free_gb(batch_dir)
            if free < MIN_FREE_GB:
                skipped += queue
                queue.clear()
                print(f"[dispatch] DISK LOW: {free:.0f} GB free is under the "
                      f"{MIN_FREE_GB} GB floor -- not starting new workers, "
                      f"{len(skipped)} instance(s) left unrun (running workers "
                      "continue). Free space and re-run to resume.", flush=True)
                break
            iid = queue.pop(0)
            cwd = batch_dir / "_cwd" / iid
            cwd.mkdir(parents=True, exist_ok=True)
            cmd = _worker_cmd(iid, batch_dir, source, agent, keep)
            running[iid] = spawn(iid, cmd, cwd, env, batch_dir / f"{iid}.log")
            print(f"[dispatch] {iid}: started ({len(running)}/{workers} slots busy, "
                  f"{len(queue)} queued)", flush=True)
        for iid in list(running):
            code = running[iid].poll()
            if code is None:
                continue
            del running[iid]
            if code == 0 and _is_complete(batch_dir, iid):
                passed = _collect_result(batch_dir, iid)["passed"]
                print(f"[dispatch] {iid}: done -> {'PASS' if passed else 'fail'}",
                      flush=True)
                done.append(iid)
            else:
                failed.append(iid)
                print(f"[dispatch] {iid}: WORKER FAILED (exit {code}), "
                      f"see {batch_dir / (iid + '.log')}", flush=True)
        if running:
            sleep(2)

    # Consolidate whatever completed; keep the ids' original order.
    results = [_collect_result(batch_dir, iid) for iid in ids if iid in done]
    write_summary(batch_dir, aggregate(results, repeat=1), results)

    # One scoreboard row for the whole batch, and only for a complete one: a
    # partial rate on the board would masquerade as a full run's number.
    # Disk-skipped instances make the batch just as incomplete as failed ones.
    if not failed and not skipped and results:
        manifest = json.loads((batch_dir / results[0]["id"] / "manifest.json")
                              .read_text(encoding="utf-8"))
        agg = aggregate(results, repeat=1)
        append_scoreboard(Path(results_root), {
            "timestamp": batch_dir.name, "agent": manifest["agent"],
            "model": manifest["model"], "source": source, "n": len(results),
            "repeat": 1, "seed": "-",
            "grading": ("official-env-corrected" if source == "swebench"
                        else "local-pytest"),
            "pass_at_1": agg["pass_at_1"], "pass_at_k": agg["pass_at_k"],
            "mean_seconds": agg["mean_seconds"], "batch_dir": str(batch_dir),
        })

    return {"results": results, "failed": failed, "skipped": skipped,
            "exit_code": 1 if failed or skipped else 0}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run eval instances in parallel worker processes.")
    p.add_argument("--batch", required=True, help="batch name under --results-root")
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--source", default="swebench", choices=["local", "swebench"])
    p.add_argument("--agent", default="ep5")
    p.add_argument("--keep", choices=["none", "failures", "all"], default="all",
                   help="passed through to run_eval; default keeps everything "
                        "(research batches need telemetry from passed runs too)")
    p.add_argument("--results-root", default="eval/results")
    args = p.parse_args(argv)

    results_root = Path(args.results_root)
    out = run_dispatch(args.ids, batch_dir=results_root / args.batch,
                       results_root=results_root, source=args.source,
                       agent=args.agent, keep=args.keep, workers=args.workers)
    agg = (aggregate(out["results"], repeat=1) if out["results"]
           else {"pass_at_1": 0.0})
    print(f"[dispatch] complete: {len(out['results'])}/{len(args.ids)} instances, "
          f"pass@1={agg['pass_at_1']:.1%}, {len(out['failed'])} worker failures, "
          f"{len(out['skipped'])} skipped (disk)", flush=True)
    return out["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
