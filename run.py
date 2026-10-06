"""
run.py — run an episode's agent as a fresh, self-contained "run".

This is the repo's harness around the episodes. Each invocation:
  1. creates a new run folder, logs/<timestamp>/ (runs never overwrite),
  2. runs the episode's agent,
  3. moves the agent's tool_calls.jsonl into that run folder,
  4. and, with --capture, also records the terminal output there (via capture.py).

Collecting runs this way makes it easy to compare how the agent's path and
tool-call count vary from one run to the next on the same task.

The episodes stay self-contained — `python agent.py` still works on its own.
run.py is just the outer harness; it can run any episode.

    python ../../run.py                    # run agent.py in the current folder
    python run.py --cwd episodes/03-loop   # run an episode from the repo root
    python ../../run.py --capture          # also record terminal output
"""
import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Our own output includes → and × in the tool-call summary; make sure stdout can
# render them on every platform (Windows consoles default to cp1252).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# A sibling module: sys.path[0] is run.py's folder, so this resolves.
import capture  # noqa: E402


def make_run_dir(logs_dir: Path) -> Path:
    """Create a fresh run folder named by the current time. If two runs start in
    the same second, add a -2, -3, ... suffix so neither is overwritten."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = logs_dir / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = logs_dir / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def plural(n, word):
    """'1 call' / '3 calls' — pluralize a word by count."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def breakdown(tools):
    """'bash, bash, read' -> 'bash×2, read×1' (preserving first-seen order)."""
    counts = {}
    for tool in tools:
        counts[tool] = counts.get(tool, 0) + 1
    return ", ".join(f"{name}×{n}" for name, n in counts.items())


def fmt(v):
    """Thousands-separate ints; leave everything else as-is."""
    return f"{v:,}" if isinstance(v, int) else str(v)


def render_single_agent(a: dict) -> None:
    """Render usage for a single-agent run (Episodes 1-5). Only the sections
    the episode actually recorded are shown — an episode with no cache, etc.
    simply omits those keys and we skip the lines."""
    print("\n=== TOKEN USAGE ===")
    print(f"agent calls:        iterations={a['iterations']}  "
          f"input={a['input_tokens']:,}  output={a['output_tokens']:,}")
    if "cache_write" in a or "cache_read" in a:
        print(f"cache:              write={a.get('cache_write', 0):,}  "
              f"read={a.get('cache_read', 0):,}")
    print(f"TOTAL:              input={a['input_tokens']:,}  "
          f"output={a['output_tokens']:,}  "
          f"grand_total={a['input_tokens'] + a['output_tokens']:,}")

    per_iter = a.get("per_iter")
    if per_iter:
        def _fmt(p):
            # dict format (Ep 3+); tolerate old in/out names too
            if isinstance(p, dict):
                mi = p.get("model_in", p.get("in"))
                mo = p.get("model_out", p.get("out"))
                to = p.get("tools_out", p.get("tool_out"))
                t = f"/{to}" if to is not None else ""
                return f"{mi}/{mo}{t}"
            return f"{p[0]}/{p[1]}"  # legacy [in, out] format
        print("per-iteration model_in/model_out/tools_out: "
              + " → ".join(_fmt(p) for p in per_iter))

    r = a.get("reasoning")
    if r:
        print("\n=== REASONING STRATEGY USAGE ===")
        print(f"write_plan calls:   {r.get('write_plan', 0)}")
        print(f"think calls:        {r.get('think', 0)}")

    s, stc = a.get("skills"), a.get("server_tool_calls")
    if s or stc:
        print("\n=== SKILLS USAGE ===")
        if s:
            print(f"list_skills calls:  {s.get('list_skills', 0)}")
            print(f"load_skill calls:   {s.get('load_skill', 0)}")
            print(f"skills loaded:      {s.get('loaded') or 'none'}")
        print(f"server-tool calls:  {stc if stc else 'none'}")


def render_multi_agent(agents: list) -> None:
    """Render usage for a multi-agent run (Episode 6): a block per worker, then
    an aggregate across all workers."""
    tot = dict(input=0, output=0, cache_w=0, cache_r=0)
    print("\n=== PER-WORKER METRICS ===")
    for a in agents:
        print(f"\n[{a['label']}]")
        print(f"  iterations:     {a['iterations']}")
        print(f"  tokens:         in={a['input_tokens']:,}  "
              f"out={a['output_tokens']:,}")
        if "cache_write" in a or "cache_read" in a:
            print(f"  cache:          write={a.get('cache_write', 0):,}  "
                  f"read={a.get('cache_read', 0):,}")
        r = a.get("reasoning")
        if r:
            print(f"  reasoning:      plan_writes={r.get('write_plan', 0)}  "
                  f"think={r.get('think', 0)}")
        s = a.get("skills")
        if s:
            print(f"  skills:         list_calls={s.get('list_skills', 0)}  "
                  f"load_calls={s.get('load_skill', 0)}  "
                  f"loaded={s.get('loaded') or 'none'}")
        if "delegate_calls" in a:
            print(f"  delegate calls: {a['delegate_calls']}")
        if a.get("server_tool_calls"):
            print(f"  server tools:   {a['server_tool_calls']}")
        tot["input"] += a["input_tokens"]
        tot["output"] += a["output_tokens"]
        tot["cache_w"] += a.get("cache_write", 0)
        tot["cache_r"] += a.get("cache_read", 0)

    print("\n=== AGGREGATE ACROSS ALL WORKERS ===")
    print(f"workers spawned:    {len(agents)}")
    print(f"total input:        {tot['input']:,}")
    print(f"total output:       {tot['output']:,}")
    print(f"total cache write:  {tot['cache_w']:,}")
    print(f"total cache read:   {tot['cache_r']:,}")
    grand = tot["input"] + tot["output"]
    print(f"grand total tokens: {grand:,}")


def print_metrics_summary(metrics_path: Path) -> None:
    """Render the usage metrics the agent recorded this run. The agent writes
    raw counters to metrics.json; the harness owns ALL reporting (same split as
    the tool-call summary). Silently does nothing if there's no file."""
    if not metrics_path.exists():
        return
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    agents = data.get("agents", [])
    if not agents:
        return

    # Echo the task so a run is self-identifying (which input produced this?).
    # The full system prompt is also recorded in metrics.json but not printed
    # here — it's long and stable; the task is the short, varying part.
    inputs = data.get("inputs") or {}
    if inputs.get("task"):
        print(f'\ninput task: "{inputs["task"]}"')

    if len(agents) == 1:
        render_single_agent(agents[0])
    else:
        render_multi_agent(agents)

    config = data.get("config")
    if config:
        print("\nconfig: " + "  ".join(f"{k}={fmt(v)}" for k, v in config.items()))


def print_tool_call_summary(tool_calls_path: Path) -> None:
    """Render a summary of the tool calls the agent recorded this run. This is
    the harness's view of the agent's telemetry — the agent writes the file but
    does not print a summary itself. Silently does nothing if there's no file."""
    if not tool_calls_path.exists():
        return
    calls = []
    with open(tool_calls_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                calls.append(json.loads(line))
    if not calls:
        return

    print("\n=== TOOL CALLS ===")
    if any("agent" in call for call in calls):
        # Multi-agent run (e.g. Episode 6): group by which agent made the call.
        per_agent = {}
        for call in calls:
            per_agent.setdefault(call.get("agent", "?"), []).append(call["tool"])
        print(f"{plural(len(calls), 'call')} "
              f"across {plural(len(per_agent), 'agent')}")
        for agent_label, tools in per_agent.items():
            print(f"  {agent_label}: {plural(len(tools), 'call')} "
                  f"— {breakdown(tools)}")
    else:
        tools = [call["tool"] for call in calls]
        print(f"{plural(len(calls), 'call')} — {breakdown(tools)}")
        print("path: " + " → ".join(tools))


def append_summary_to_capture(run_dir: Path, text: str) -> None:
    """The tool-call and usage summaries render after the capture window has
    closed (they need the collected files), so a captured run's terminal.log
    would end at the agent's final response and the end-of-run summary would
    be missing from the recording. Append the same lines to both capture
    files, stamped with the run's end time, so the logs end the way the live
    terminal did."""
    log_path = run_dir / "terminal.log"
    jsonl_path = run_dir / "terminal.jsonl"
    if not log_path.exists():
        return
    duration = 0.0
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == "end":
                    duration = record.get("duration_s", 0.0)
    with open(log_path, "a", encoding="utf-8") as f:
        for line in text.splitlines():
            f.write(f"[+{duration:>7.2f}s] {line}\n")
    if jsonl_path.exists():
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for line in text.splitlines():
                f.write(json.dumps({"t": duration, "text": line}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an episode's agent as a fresh, recorded run.")
    parser.add_argument("--cwd", default=".",
                        help="episode directory to run in (default: current)")
    parser.add_argument("--logdir", default=None,
                        help="where run folders are created (default: <cwd>/logs)")
    parser.add_argument("--capture", action="store_true",
                        help="also record terminal output into the run folder")
    parser.add_argument(
        "-g", "--grade", action="store_true",
        help="after the run, run the episode's grade.py and record its output "
             "in the run folder (grade.log)",
    )
    parser.add_argument(
        "--collect", action="append", default=None,
        help="agent output file(s) to move into the run folder "
             "(default: tool_calls.jsonl, metrics.json, messages.jsonl)",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="command to run; defaults to `python -u agent.py`",
    )
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    command = capture.resolve_command(args.command)
    logs_dir = Path(args.logdir).resolve() if args.logdir else cwd / "logs"
    collect = args.collect if args.collect is not None else [
        "tool_calls.jsonl", "metrics.json", "messages.jsonl",
    ]

    run_dir = make_run_dir(logs_dir)
    print(f"[run] {' '.join(command)}  ->  {run_dir}\n", flush=True)

    if args.capture:
        # capture.py records the terminal into the run folder (and mirrors live).
        exit_code = capture.run_and_capture(command, cwd, run_dir)
    else:
        # No capture: let the command print straight to this terminal.
        exit_code = subprocess.run(command, cwd=cwd).returncode

    # Move the agent's own output files into this run's folder, so a fixed-name
    # file like tool_calls.jsonl is preserved per run instead of being overwritten.
    for name in collect:
        produced = cwd / name
        if produced.exists():
            shutil.move(str(produced), str(run_dir / name))
            print(f"[run] collected {name}", flush=True)

    # Keep the sandbox as the agent left it, too: the next run wipes it, and the
    # work itself (not only the log of making it) is what a comparison needs.
    sandbox = cwd / "sandbox"
    if sandbox.exists():
        shutil.copytree(sandbox, run_dir / "sandbox",
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        print("[run] collected sandbox/", flush=True)

    # The harness — not the agent — renders the tool-call and usage summaries.
    # Rendered into a buffer so a captured run can also append them to its
    # terminal logs (they print after the capture window closes).
    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        print_tool_call_summary(run_dir / "tool_calls.jsonl")
        print_metrics_summary(run_dir / "metrics.json")
    print(summary.getvalue(), end="", flush=True)
    if args.capture:
        append_summary_to_capture(run_dir, summary.getvalue())

    # --grade: run the episode's own grading script (held-out tests) against
    # the sandbox the run left behind, and keep the verdict with the run.
    # Grading is judgment on the finished run — it never feeds back into the
    # agent, so it runs strictly after everything above.
    if args.grade:
        grade_script = cwd / "grade.py"
        if not grade_script.exists():
            # Not every episode has a grading layer; --grade just skips then.
            print("[run] --grade: no grade.py in this episode, skipping", flush=True)
        else:
            graded = subprocess.run(
                [sys.executable, "grade.py"], cwd=cwd,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            grade_output = graded.stdout
            if graded.stderr.strip():
                grade_output += "\n" + graded.stderr
            end = "" if grade_output.endswith("\n") else "\n"
            print(grade_output, end=end, flush=True)
            (run_dir / "grade.log").write_text(grade_output, encoding="utf-8")
            # Also append the verdict to the captured terminal files (same as the
            # end-of-run summaries), so terminal.log tells the whole story and
            # nobody has to open grade.log to learn whether the run passed.
            append_summary_to_capture(run_dir, grade_output)
            print(f"[run] grade recorded -> {run_dir / 'grade.log'}", flush=True)
            if exit_code == 0:
                exit_code = graded.returncode

    print(f"\n[run] done (exit {exit_code}) -> {run_dir}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
