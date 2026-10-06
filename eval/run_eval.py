"""CLI + orchestration for the eval harness.

    python -m eval.run_eval --source local --n 5 --repeat 3
    python -m eval.run_eval --source local --id md2html__alerts
    python -m eval.run_eval --source local --agent fake-fixing --n 1
        # token-free smoke
    python -m eval.run_eval --source swebench --n 5
        # grades each sample as it lands (--no-grade to skip)
    python -m eval.run_eval --source swebench --n 20 --stratified
        # proportional easy/medium/hard mix

The agent-under-test is selectable: `--agent ep5` (default, the real reference
agent) or a fake adapter for token-free testing. Providers supply instances;
sampling picks which to run; the runner scores each; results.py records them.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from eval.fake_agents import fixing_solver, noop_solver
from eval.results import (aggregate, append_scoreboard, apply_retention,
                          write_manifest, write_summary)
from eval.runner import run_instance
from eval.sampling import filter_pool, sample, stratified_sample

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_local_instances():
    from eval.targets.local import get_instances
    return get_instances()


def _load_instances(source):
    if source == "local":
        return load_local_instances()
    if source == "swebench":
        from eval.targets.swebench import get_instances
        return get_instances()  # deferred stub: raises with a clear message
    raise SystemExit(f"unknown source: {source}")


def _select_agent(name):
    # The model label is derived from the agent's own config (eval.agent.MODEL),
    # not a CLI flag: a free-text flag only labels the manifest and silently
    # drifts from what the agent actually calls.
    if name == "fake-fixing":
        return fixing_solver, "fake-fixing", "(fake)"
    if name == "fake-noop":
        return noop_solver, "fake-noop", "(fake)"
    if name == "ep5":
        # imported lazily: only the real run needs it
        from eval.agent import MODEL, solve
        return solve, "ep5", MODEL
    raise SystemExit(f"unknown agent: {name}")


def _grade_now(inst_dir, model_label) -> bool:
    """--grade path: officially grade one just-finished sample. A grading
    failure is FATAL by design: --grade exists to surface a broken setup at
    the first sample, and scoring an infra failure as an agent failure would
    corrupt the pass rate. Finished samples stay on disk; after fixing the
    environment, grade them with eval.official and re-run what's missing."""
    from eval.official import grade_instance  # lazy: needs WSL + Docker only here
    verdict = grade_instance(inst_dir, model_label)
    state = "RESOLVED" if verdict["resolved"] else "unresolved"
    print(f"[eval] {inst_dir.name}: official verdict {state}", flush=True)
    return verdict["resolved"]


def _cleanup_image(instance_id) -> None:
    """--clean-images path: drop the instance's image once its sample is fully
    done, so a batch's disk footprint stays at the in-flight set instead of
    growing with every sample (the 2026-07-03 batch died of a full disk). The
    next same-repo instance may re-pull shared layers -- bandwidth, not disk.
    A failed removal is reported and tolerated: it only costs disk."""
    from eval import container  # lazy: mirrors _grade_now, needs Docker only here
    ok = container.remove_image(instance_id)
    state = "removed" if ok else "removal FAILED (image kept)"
    print(f"[eval] {instance_id}: instance image {state}", flush=True)


def _agent_git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run the reference agent over a pool of problems.")
    p.add_argument("--source", default="local", choices=["local", "swebench"])
    p.add_argument("--agent", default="ep5", help="ep5 | fake-fixing | fake-noop")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--id", dest="instance_id", default=None)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None,
                   help="filter by Verified's time-to-fix bucket before sampling")
    p.add_argument("--repo", default=None,
                   help="filter by repo substring, e.g. flask")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratified", action="store_true",
                   help="spread --n across the difficulty buckets in proportion "
                        "to their share of the pool (instead of one flat random "
                        "draw), so a small sample mirrors the benchmark's mix. "
                        "Needs difficulty metadata, so swebench only.")
    p.add_argument("--grade", action=argparse.BooleanOptionalAction, default=None,
                   help="officially grade each sample right after it runs (solve -> "
                        "grade -> next sample), so a broken setup surfaces after the "
                        "first sample, not after the whole batch. Default: ON for "
                        "swebench (--no-grade opts out); unavailable for local.")
    p.add_argument("--clean-images", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="remove each instance's Docker image once the sample is "
                        "fully done (solved + graded), so batch disk use stays at "
                        "the in-flight set instead of growing per sample. Default: "
                        "ON for swebench (--no-clean-images keeps images, e.g. to "
                        "re-enter a container while debugging); "
                        "unavailable for local.")
    p.add_argument("--keep", choices=["none", "failures", "all"], default="failures")
    p.add_argument("--results-root", default="eval/results")
    p.add_argument("--timestamp", default=None,
                   help="override batch id (tests use this)")
    args = p.parse_args(argv)

    solve, agent_label, model_label = _select_agent(args.agent)
    if args.stratified and args.difficulty:
        raise SystemExit("--stratified spans the difficulty buckets; it can't "
                         "be combined with a --difficulty filter")
    if args.stratified and args.instance_id:
        raise SystemExit("--stratified samples a pool; it can't be combined "
                         "with --id")
    if args.grade and args.source != "swebench":
        raise SystemExit("--grade needs --source swebench "
                         "(official grading only exists there)")
    # unset: official verdicts are the default where they exist
    if args.grade is None:
        args.grade = args.source == "swebench"
    if args.clean_images and args.source != "swebench":
        raise SystemExit("--clean-images needs --source swebench "
                         "(only its instances have images)")
    if args.clean_images is None:  # unset: cleanup is the default where images exist
        args.clean_images = args.source == "swebench"
    instances = filter_pool(_load_instances(args.source),
                            difficulty=args.difficulty, repo=args.repo)
    if args.stratified:
        try:
            picked = stratified_sample(instances, n=args.n, seed=args.seed)
        except ValueError as e:
            raise SystemExit(str(e))
    else:
        picked = sample(instances, n=args.n, seed=args.seed,
                        instance_id=args.instance_id)
    if not picked:
        raise SystemExit(
            "no instances selected (check --id / --difficulty / --repo)")

    results_root = Path(args.results_root)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = results_root / stamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    # The manifest is known before anything runs; writing it (and rewriting the
    # summary after every sample) means a fatal grader/setup failure mid-batch
    # still leaves a record of what ran and how it scored so far.
    write_manifest(batch_dir, {
        "timestamp": stamp, "agent": agent_label, "agent_git_sha": _agent_git_sha(),
        "model": model_label, "source": args.source, "n": args.n,
        "repeat": args.repeat,
        "seed": args.seed, "stratified": args.stratified,
        "filters": {"difficulty": args.difficulty, "repo": args.repo},
        "instance_ids": [i.id for i in picked],
    })

    def _score_of_record(rows):
        # With --grade the official verdicts are the numbers of record; without
        # it, swebench rows stay explicitly ungraded (local pytest is env-noise).
        return ([dict(r, passed=r["official_resolved"]) for r in rows]
                if args.grade else rows)

    results = []
    for inst in picked:
        for k in range(args.repeat):
            label = inst.id if k == 0 else f"{inst.id}-run{k + 1}"
            print(f"[eval] {label}", flush=True)
            result = run_instance(inst, solve, batch_dir, run_label=label)
            if args.grade:
                result["official_resolved"] = _grade_now(batch_dir / label,
                                                         model_label)
            results.append(result)
            scored = _score_of_record(results)
            write_summary(batch_dir, aggregate(scored, repeat=args.repeat), scored)
        # After ALL repeats of this instance: earlier removal would force the
        # next repeat to re-pull the image it just deleted.
        if args.clean_images:
            _cleanup_image(inst.id)

    results = _score_of_record(results)
    agg = aggregate(results, repeat=args.repeat)
    write_summary(batch_dir, agg, results)
    append_scoreboard(results_root, {
        "timestamp": stamp, "agent": agent_label, "model": model_label,
        "source": args.source,
        "n": len(picked), "repeat": args.repeat, "seed": args.seed,
        # Local pytest is only meaningful for the local provider; swebench rows
        # are ungraded unless --grade ran official grading per sample.
        "grading": ("official-env-corrected" if args.grade
                    else "local-pytest" if args.source == "local" else "ungraded"),
        "pass_at_1": agg["pass_at_1"], "pass_at_k": agg["pass_at_k"],
        "mean_seconds": agg["mean_seconds"], "batch_dir": str(batch_dir),
    })
    apply_retention(batch_dir, results, keep=args.keep)

    print(f"[eval] pass@1={agg['pass_at_1']:.1%}  pass@k={agg['pass_at_k']:.1%}  "
          f"-> {batch_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
