"""grade.py — grade a finished run against the held-out tests.

The agent verifies its own work while it runs (it has the visible fixture and
whatever tests it writes). Grading is different: a judgment rendered AFTER the
run, against tests the agent never saw. The held-out tests live in held_out/
(outside initial/, so the sandbox reset never copies them in); this script
injects them into the sandbox and runs the full suite.

    python grade.py          # run after agent.py has finished

A run passes the grade when the whole suite is green WITH the held-out tests
included. Failing here while the agent's own run looked green is the
interesting case: it means the implementation fit the examples it could see
but not the rule they came from.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
HELD_OUT = HERE / "held_out"
SANDBOX = HERE / "sandbox"


def main() -> int:
    if not SANDBOX.exists():
        print("No sandbox/ found - run the agent first (python agent.py).")
        return 2

    # Inject: copy every held-out file to the same relative path in the sandbox.
    injected = []
    for src in HELD_OUT.rglob("*.py"):
        rel = src.relative_to(HELD_OUT)
        dest = SANDBOX / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        injected.append(str(rel))
    print(f"Injected {len(injected)} held-out file(s): {', '.join(injected)}\n")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=SANDBOX,
    )
    print()
    print("GRADE: PASS" if result.returncode == 0 else "GRADE: FAIL")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
