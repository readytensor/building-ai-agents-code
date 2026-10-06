"""
capture.py — run a command and tee its output to a log.

Single responsibility: run a command, mirror its output to your terminal live,
and also write a copy (each line stamped with elapsed time) to a log directory.
It does NOT create runs, manage timestamps, or move other files — wrapping a
"run" around a command is a separate concern (see run.py).

Cross-platform and project-agnostic. Use it from the command line, or import it
and call run_and_capture() (which is what run.py does).

Command line:
    python capture.py -- pytest -q
    python capture.py --out mylogs -- python script.py

Writes into <out> (default: the current directory):
    terminal.log     human-readable, each line prefixed [+12.34s]
    terminal.jsonl   one JSON object per line: {"t": 12.34, "text": "..."}
                     plus a leading "meta" record and a trailing "end" record.

The command is run unbuffered (PYTHONUNBUFFERED) so the live mirror and the log
stay in real time and in order.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_and_capture(command, cwd, out_dir) -> int:
    """Run `command` in `cwd`, mirror its output to this terminal live, and tee
    a copy into out_dir/terminal.log and out_dir/terminal.jsonl. Returns the
    command's exit code."""
    cwd = Path(cwd)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plain_log = out_dir / "terminal.log"
    json_log = out_dir / "terminal.jsonl"

    # Force the child to emit output unbuffered and in UTF-8, so the live mirror
    # and the saved log stay in step (and arrows/checkmarks survive).
    child_env = dict(os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    plain_file = open(plain_log, "w", encoding="utf-8")
    json_file = open(json_log, "w", encoding="utf-8")

    def write_json_record(record: dict) -> None:
        json_file.write(json.dumps(record) + "\n")
        json_file.flush()

    write_json_record({
        "type": "meta",
        "command": command,
        "cwd": str(cwd),
        "started": datetime.now().isoformat(timespec="seconds"),
    })

    start_time = time.monotonic()
    exit_code = 0
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=child_env,
        stdout=subprocess.PIPE,     # we capture stdout, and fold stderr into it,
        stderr=subprocess.STDOUT,   # so errors are logged too
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,                  # line-buffered on our side
    )
    try:
        # Read the command's output one line at a time, as it appears.
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            elapsed = round(time.monotonic() - start_time, 2)
            print(line, flush=True)                                  # live mirror
            plain_file.write(f"[+{elapsed:>7.2f}s] {line}\n")        # readable log
            plain_file.flush()
            write_json_record({"t": elapsed, "text": line})          # structured log
        exit_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        exit_code = process.wait()
        print("\n[capture] interrupted", flush=True)

    duration = round(time.monotonic() - start_time, 2)
    write_json_record({"type": "end", "exit_code": exit_code, "duration_s": duration})
    plain_file.close()
    json_file.close()
    return exit_code


def resolve_command(raw_command):
    """Turn leftover command-line args into a command list. argparse leaves a
    leading `--` in place, so drop it. With nothing given, default to running
    agent.py with the current interpreter."""
    command = list(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [sys.executable, "-u", "agent.py"]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and tee its output to a log.")
    parser.add_argument("--cwd", default=".",
                        help="working directory to run in (default: current)")
    parser.add_argument("--out", default=".",
                        help="directory to write terminal.log/.jsonl into "
                             "(default: current)")
    parser.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="command to run; defaults to `python -u agent.py`",
    )
    args = parser.parse_args()

    # The output we mirror may contain unicode (→ ✓ etc.); make sure our own
    # stdout can render it on every platform (Windows consoles default to cp1252).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    command = resolve_command(args.command)
    cwd = Path(args.cwd).resolve()
    out_dir = Path(args.out).resolve()

    print(f"[capture] running: {' '.join(command)}  ->  {out_dir}\n", flush=True)
    exit_code = run_and_capture(command, cwd, out_dir)
    print(f"\n[capture] done (exit {exit_code}) -> {out_dir}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
