"""Batch results: aggregate per-instance outcomes, write the human/machine
summaries and manifest, append the cross-run scoreboard, and enforce log
retention so results never balloon.

pass@1 = fraction of instances whose FIRST attempt passed.
pass@k = fraction of instances that passed on AT LEAST ONE of their attempts.
"""
import gzip
import json
import shutil
from pathlib import Path


def aggregate(results: list, repeat: int) -> dict:
    """Collapse per-attempt results (repeat attempts per instance) into rates."""
    by_id = {}
    for r in results:
        by_id.setdefault(r["id"], []).append(r)
    n = len(by_id)
    first_pass = sum(1 for attempts in by_id.values() if attempts[0]["passed"])
    any_pass = sum(1 for attempts in by_id.values()
                   if any(a["passed"] for a in attempts))
    seconds = [r["seconds"] for r in results]
    return {
        "n_instances": n,
        "repeat": repeat,
        "pass_at_1": round(first_pass / n, 4) if n else 0.0,
        "pass_at_k": round(any_pass / n, 4) if n else 0.0,
        "mean_seconds": round(sum(seconds) / len(seconds), 3) if seconds else 0.0,
    }


def write_summary(batch_dir: Path, agg: dict, results: list) -> None:
    (batch_dir / "summary.json").write_text(
        json.dumps({"aggregate": agg, "instances": results}, indent=2),
        encoding="utf-8"
    )
    lines = [
        "# Eval batch summary", "",
        f"- instances: {agg['n_instances']}  (repeat={agg['repeat']})",
        f"- pass@1: {agg['pass_at_1']:.1%}",
        f"- pass@k: {agg['pass_at_k']:.1%}",
        f"- mean seconds/attempt: {agg['mean_seconds']}", "",
        "| instance | attempt | passed | seconds |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['id']} | {r['run_label']} | "
                     f"{'PASS' if r['passed'] else 'fail'} | {r['seconds']} |")
    (batch_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(batch_dir: Path, manifest: dict) -> None:
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                             encoding="utf-8")


_SCOREBOARD_COLS = ["timestamp", "agent", "model", "source", "n", "repeat",
                    "seed", "grading", "pass_at_1", "pass_at_k", "mean_seconds",
                    "batch_dir"]


def append_scoreboard(results_root: Path, row: dict) -> None:
    """Append one batch row to scoreboard.jsonl and re-render scoreboard.md."""
    results_root.mkdir(parents=True, exist_ok=True)
    jsonl = results_root / "scoreboard.jsonl"
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    rows = [json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    header = "| " + " | ".join(_SCOREBOARD_COLS) + " |"
    sep = "|" + "|".join(["---"] * len(_SCOREBOARD_COLS)) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in _SCOREBOARD_COLS) + " |"
            for r in rows]
    (results_root / "scoreboard.md").write_text(
        "\n".join([header, sep, *body]) + "\n", encoding="utf-8")


# --- Retention: keep verbose logs only where you'd actually look (failures). ---
# metrics.json and final_message.md are deliberately NOT here: both are small
# and both proved necessary for analyzing PASSED runs too (batch20 lost the
# telemetry of every resolved sample to this list).
_VERBOSE = ("output.log", "tool_calls.jsonl", "transcript.json", "messages.jsonl")


def apply_retention(batch_dir: Path, results: list, keep: str) -> None:
    """keep='all' keeps everything; 'none' drops all verbose logs; 'failures'
    (default) keeps verbose logs only for failed attempts, gzipped."""
    if keep == "all":
        return
    for r in results:
        inst_dir = Path(r["inst_dir"])
        drop = keep == "none" or r["passed"]
        for name in _VERBOSE:
            path = inst_dir / name
            if not path.exists():
                continue
            if drop:
                path.unlink()
            else:
                with path.open("rb") as src, \
                        gzip.open(str(path) + ".gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                path.unlink()
