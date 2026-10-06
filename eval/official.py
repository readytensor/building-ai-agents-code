"""Official SWE-bench grading, integrated: batch dir in, env-corrected verdicts out.

    python -m eval.official eval/results/<batch> --model-name ep5-haiku

Flow:
  1. write predictions.jsonl from the batch's diff.patch files (first attempt
     per instance -- the official grader keys on instance_id).
  2. ensure a GOLD BASELINE exists for every instance: grade the maintainers'
     own patch once per instance and cache which tests fail anyway in THIS
     environment (e.g. network-dependent tests under Docker Desktop/WSL2).
     A verdict measured without this baseline silently converts environment
     quirks into agent failures.
  3. grade the agent predictions (WSL + Docker via eval/wsl/grade.sh).
  4. score env-corrected: resolved = all FAIL_TO_PASS pass AND no PASS_TO_PASS
     failures beyond what gold itself fails here. Write official.json per
     instance and append a scoreboard row.

The WSL/Docker invocation is injectable (`runner=`), so the scoring logic is
fully tested offline; only the thin bridge needs the real environment.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from eval.results import append_scoreboard

_EVAL_DIR = Path(__file__).resolve().parent
GOLD_BASELINE_DIR = _EVAL_DIR / "cache" / "gold_baselines"
_GRADE_SH = _EVAL_DIR / "wsl" / "grade.sh"
_WSL_DISTRO = "Ubuntu-24.04"

# Repeat attempts get "-runN" directory suffixes (run_eval's run_label); the
# one rule for recognizing them, shared by everything that maps an attempt
# directory back to its instance id.
_RUN_SUFFIX = re.compile(r"-run\d+$")


def wsl_path(win_path: Path) -> str:
    """C:\\Users\\... -> /mnt/c/Users/... (forward slashes, lowercase drive)."""
    p = Path(win_path).resolve()
    drive = p.drive.rstrip(":").lower()
    rest = "/".join(p.parts[1:])
    return f"/mnt/{drive}/{rest}"


def run_grader(predictions: str, run_id: str, out_dir: Path, ids: list) -> None:
    """The real bridge: run grade.sh inside WSL. `predictions` is either a
    predictions.jsonl path (Windows) or the literal string "gold"."""
    pred_arg = predictions if predictions == "gold" else wsl_path(Path(predictions))
    cmd = ["wsl", "-d", _WSL_DISTRO, "--", "bash", wsl_path(_GRADE_SH),
           pred_arg, run_id, wsl_path(Path(out_dir)), *ids]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"grading failed (exit {proc.returncode}):\n"
                           f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def write_predictions(batch_dir: Path, model_name: str) -> Path:
    """predictions.jsonl from the batch's first attempt per instance (repeat
    attempts get -runN dir suffixes; the grader keys on instance_id)."""
    preds = []
    for d in sorted(Path(batch_dir).iterdir()):
        if (d.is_dir() and (d / "diff.patch").exists()
                and not _RUN_SUFFIX.search(d.name)):
            preds.append({
                "instance_id": d.name,
                "model_name_or_path": model_name,
                "model_patch": (d / "diff.patch").read_text(encoding="utf-8"),
            })
    path = Path(batch_dir) / "predictions.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    return path


def parse_report(report_path: Path, instance_id: str) -> dict:
    """Reduce one official report.json to the fields scoring needs."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8-sig"))[instance_id]
    tests = data["tests_status"]
    return {
        "resolved_raw": bool(data.get("resolved")),
        "f2p_ok": not tests["FAIL_TO_PASS"]["failure"],
        "p2p_failures": sorted(tests["PASS_TO_PASS"]["failure"]),
    }


def env_corrected(agent: dict, gold: dict) -> bool:
    """Resolved, measured against this machine: every F2P passes and the agent
    fails nothing that gold does not also fail here."""
    return agent["f2p_ok"] and set(agent["p2p_failures"]) <= set(gold["p2p_failures"])


def _official_verdict(agent: dict, gold: dict) -> dict:
    """The official.json payload for one graded attempt."""
    return {
        # env-corrected, the number that counts
        "resolved": env_corrected(agent, gold),
        "resolved_raw": agent["resolved_raw"],   # the grader's uncorrected verdict
        "f2p_ok": agent["f2p_ok"],
        "p2p_failures": agent["p2p_failures"],
        "gold_baseline_failures": gold["p2p_failures"],
        "beyond_gold": sorted(set(agent["p2p_failures"]) - set(gold["p2p_failures"])),
    }


def _empty_patch_verdict() -> dict:
    """An empty patch is a legitimate agent outcome (it finished without
    changing anything), not an infra failure -- but the official harness skips
    empty predictions without writing a report. Grade it directly: it can't
    pass FAIL_TO_PASS, so it is unresolved by definition."""
    return {
        "resolved": False, "resolved_raw": False, "f2p_ok": False,
        "p2p_failures": [], "gold_baseline_failures": [], "beyond_gold": [],
        "note": "empty patch: the agent made no changes; graded unresolved "
                "without running the harness",
    }


def _gold_baselines(instance_ids: list, run_id: str, runner) -> dict:
    """Load cached gold baselines; grade the gold patch for any missing ones."""
    GOLD_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baselines, missing = {}, []
    for iid in instance_ids:
        cached = GOLD_BASELINE_DIR / f"{iid}.json"
        if cached.exists():
            baselines[iid] = json.loads(cached.read_text(encoding="utf-8"))
        else:
            missing.append(iid)
    if missing:
        out = GOLD_BASELINE_DIR / "_reports"
        runner("gold", f"{run_id}-gold", out, missing)
        for iid in missing:
            baselines[iid] = parse_report(out / f"{iid}.json", iid)
            (GOLD_BASELINE_DIR / f"{iid}.json").write_text(
                json.dumps(baselines[iid], indent=2), encoding="utf-8")
    return baselines


def grade_instance(inst_dir, model_name: str, run_id: str = None,
                   runner=run_grader) -> dict:
    """Officially grade ONE finished attempt (inst_dir holds its diff.patch).

    This is the solve -> grade -> next-sample path (run_eval --grade): a broken
    setup surfaces after the first sample instead of after a costly full batch.
    Writes official.json into inst_dir and returns it."""
    inst_dir = Path(inst_dir)
    # The grader keys on the real instance id, not the attempt label.
    iid = _RUN_SUFFIX.sub("", inst_dir.name)
    # The run_id must be unique per invocation: the harness reuses any report
    # already sitting in the run_id's WSL workdir, so a recycled id silently
    # returns a PREVIOUS attempt's verdict (this happened: a killed batch and
    # its rerun shared batch names, and the rerun inherited stale verdicts).
    run_id = run_id or (f"{inst_dir.parent.name}-{inst_dir.name}-"
                        f"{time.strftime('%m%d%H%M%S')}").replace(".", "-")

    patch = (inst_dir / "diff.patch").read_text(encoding="utf-8")
    if not patch.strip():
        verdict = _empty_patch_verdict()
        (inst_dir / "official.json").write_text(json.dumps(verdict, indent=2),
                                                encoding="utf-8")
        return verdict

    pred = {"instance_id": iid, "model_name_or_path": model_name,
            "model_patch": patch}
    predictions = inst_dir / "predictions.jsonl"
    predictions.write_text(json.dumps(pred) + "\n", encoding="utf-8")

    gold = _gold_baselines([iid], run_id, runner)
    reports_dir = inst_dir / "official_reports"
    runner(str(predictions), run_id, reports_dir, [iid])

    verdict = _official_verdict(parse_report(reports_dir / f"{iid}.json", iid),
                                gold[iid])
    (inst_dir / "official.json").write_text(json.dumps(verdict, indent=2),
                                            encoding="utf-8")
    return verdict


def grade_batch(batch_dir, model_name: str, run_id: str = None,
                runner=run_grader) -> dict:
    batch_dir = Path(batch_dir)
    run_id = run_id or (f"{batch_dir.name}-"
                        f"{time.strftime('%m%d%H%M%S')}").replace(".", "-")
    predictions = write_predictions(batch_dir, model_name)
    preds = [json.loads(line)
             for line in predictions.read_text(encoding="utf-8").splitlines()]
    ids = [p["instance_id"] for p in preds]
    if not ids:
        raise SystemExit(f"no diff.patch files found under {batch_dir}")
    # Empty patches never reach the harness (it writes no report for them);
    # they grade directly as unresolved.
    empty = {p["instance_id"] for p in preds if not p["model_patch"].strip()}
    graded_ids = [i for i in ids if i not in empty]

    gold = _gold_baselines(graded_ids, run_id, runner)

    reports_dir = batch_dir / "official_reports"
    if graded_ids:
        runner(str(predictions), run_id, reports_dir, graded_ids)

    resolved, unresolved = [], []
    for iid in ids:
        verdict = (_empty_patch_verdict() if iid in empty else
                   _official_verdict(parse_report(reports_dir / f"{iid}.json", iid),
                                     gold[iid]))
        (resolved if verdict["resolved"] else unresolved).append(iid)
        (batch_dir / iid / "official.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id, "model": model_name,
        "resolved": resolved, "unresolved": unresolved,
        "official_pass_at_1": round(len(resolved) / len(ids), 4),
    }
    (batch_dir / "official_summary.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    append_scoreboard(batch_dir.parent, {
        "timestamp": batch_dir.name, "agent": "ep5", "model": model_name,
        "source": "swebench", "n": len(ids), "grading": "official-env-corrected",
        "pass_at_1": summary["official_pass_at_1"], "batch_dir": str(batch_dir),
    })
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Officially grade an eval batch (WSL + Docker).")
    p.add_argument("batch_dir")
    p.add_argument("--model-name", required=True,
                   help="model_name_or_path for predictions.jsonl")
    p.add_argument("--run-id", default=None)
    args = p.parse_args(argv)
    summary = grade_batch(args.batch_dir, args.model_name, run_id=args.run_id)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
