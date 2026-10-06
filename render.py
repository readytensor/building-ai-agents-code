"""
render.py — render a Markdown file with an episode's md2html, to see what the
agent built.

Single responsibility: run md2html on one Markdown file and write the HTML page
next to it. After a run, the agent's version of md2html is in the episode's
sandbox/, and this is how you try it on a real document.

Which md2html it uses matters. An `md2html` command on your PATH is whatever copy
pip installed, not the one the agent just changed, so this script runs md2html
from inside the folder you name and always uses that copy.

Command line (from the repo root):

    # the agent's md2html, after a run
    python render.py --cwd episodes/04-tools examples/emoji.md
    # the finished md2html in examples/
    python render.py examples/about-the-series.md
    # also open the page in a browser
    python render.py --cwd episodes/03-loop examples/toc.md --open

Writes <input>.html next to the input: a complete page with md2html's built-in
stylesheet (its --standalone output).
"""
import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINISHED = ROOT / "examples" / "md2html"


def shown(path: Path) -> str:
    """A path as the user would type it: relative to the repo root when inside it."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def project_dir(episode) -> Path:
    """The folder whose md2html we run: the episode's sandbox, or the finished one."""
    if episode is None:
        return FINISHED
    sandbox = Path(episode).resolve() / "sandbox"
    if not (sandbox / "md2html").is_dir():
        raise SystemExit(f"[render] no md2html in {shown(sandbox)}: run the agent "
                         f"first (python run.py --cwd {episode})")
    return sandbox


def render(source: Path, project: Path) -> Path:
    """Run `python -m md2html` from inside `project`, so `import md2html` finds that
    copy first, and write the standalone page next to the source. Returns its path."""
    output = source.with_suffix(".html")
    child_env = dict(os.environ)
    # PYTHONSAFEPATH would drop the project folder from sys.path
    child_env.pop("PYTHONSAFEPATH", None)
    child_env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", "md2html", str(source),
         "--standalone", "-o", str(output)],
        cwd=project, env=child_env,
    )
    if result.returncode != 0:
        raise SystemExit(f"[render] md2html failed (exit {result.returncode})")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a Markdown file with an episode's md2html.")
    parser.add_argument("input",
                        help="the .md file to render, e.g. examples/emoji.md")
    parser.add_argument("--cwd", metavar="EPISODE",
                        help="an episode folder; uses the md2html in its sandbox/ "
                             "(default: the finished one in examples/md2html)")
    parser.add_argument("--open", action="store_true",
                        help="open the page in a browser")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    source = Path(args.input).resolve()
    if not source.is_file():
        raise SystemExit(f"[render] no such file: {args.input}")
    project = project_dir(args.cwd)
    output = render(source, project)
    print(f"[render] md2html from {shown(project)} -> {shown(output)}", flush=True)
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
