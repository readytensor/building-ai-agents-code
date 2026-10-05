"""md2html CLI entry point.

Usage:

    md2html INPUT_FILE [-o OUTPUT_FILE] [--stdout] [-s]
            [--no-extensions] [--extensions LIST]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import __version__, render
from .extensions import available_extensions, default_extensions, resolve_extensions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="md2html",
        description="Convert a Markdown file to HTML.",
    )
    p.add_argument("input", metavar="INPUT_FILE", help="Path to a .md file.")
    out = p.add_mutually_exclusive_group()
    out.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT_FILE",
        help="Output .html path (default: replace .md with .html).",
    )
    out.add_argument(
        "--stdout",
        action="store_true",
        help="Write to stdout instead of a file.",
    )
    ext = p.add_mutually_exclusive_group()
    ext.add_argument(
        "--no-extensions",
        action="store_true",
        help="Disable all extensions (core markdown only).",
    )
    ext.add_argument(
        "--extensions",
        metavar="LIST",
        help=(
            "Comma-separated list of extensions to enable "
            f"(available: {', '.join(available_extensions())})."
        ),
    )
    p.add_argument(
        "-s",
        "--standalone",
        action="store_true",
        help=(
            "Wrap the output in a complete, standalone HTML document "
            "(with a built-in stylesheet) instead of just the body fragment."
        ),
    )
    p.add_argument("--version", action="version", version=f"md2html {__version__}")
    return p


# Minimal built-in stylesheet for --standalone output. Inlined so the page is
# self-contained (no external CSS to ship). GitHub-ish; also styles the alert
# classes the github_alerts extension emits.
_STYLE = """\
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       line-height: 1.6; max-width: 720px; margin: 2rem auto; padding: 0 1.25rem; color: #1f2328; }
h1, h2, h3 { line-height: 1.25; margin-top: 1.8rem; }
h1, h2 { border-bottom: 1px solid #d0d7de; padding-bottom: .3rem; }
a { color: #0969da; }
code { background: #eff1f3; padding: .15em .35em; border-radius: 4px; font-size: 90%; }
pre { background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow: auto; }
pre code { background: none; padding: 0; font-size: 100%; }
blockquote { margin: 1rem 0; padding: 0 1rem; color: #59636e; border-left: .25rem solid #d0d7de; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #d0d7de; padding: .4rem .6rem; text-align: left; }
th { background: #f6f8fa; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 2rem 0; }
del { color: #59636e; }
.markdown-alert { padding: .6rem 1rem; margin: 1rem 0; border-left: .25rem solid #d0d7de;
                  border-radius: 6px; background: #f6f8fa; }
.markdown-alert-title { font-weight: 600; margin: 0 0 .3rem; text-transform: capitalize; }
.markdown-alert-note { border-left-color: #0969da; }
.markdown-alert-tip { border-left-color: #1a7f37; }
.markdown-alert-important { border-left-color: #8250df; }
.markdown-alert-warning { border-left-color: #9a6700; }
.markdown-alert-caution { border-left-color: #cf222e; }
"""


def _standalone_document(fragment: str, title: str) -> str:
    """Wrap a body fragment in a minimal, self-contained HTML5 page.

    The library deliberately emits only the body fragment (the idiomatic
    contract for a Markdown converter — the caller owns the page shell). The
    CLI's --standalone flag adds the document scaffolding + a small built-in
    stylesheet so the output opens cleanly in a browser on its own.
    """
    from .utils import html_escape

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html_escape(title)}</title>\n"
        f"<style>\n{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{fragment}\n"
        "</body>\n"
        "</html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"md2html: input file not found: {in_path}", file=sys.stderr)
        return 2
    source = in_path.read_text(encoding="utf-8")

    if args.no_extensions:
        exts = []
    elif args.extensions:
        try:
            exts = resolve_extensions(args.extensions)
        except ValueError as e:
            print(f"md2html: {e}", file=sys.stderr)
            return 2
    else:
        exts = default_extensions()

    html = render(source, extensions=exts)

    if args.standalone:
        # Title: the first level-1 heading if present, else the file name.
        m = re.search(r"^\s{0,3}#\s+(.+?)\s*$", source, re.MULTILINE)
        title = m.group(1).strip() if m else in_path.stem
        html = _standalone_document(html, title)

    if args.stdout:
        sys.stdout.write(html)
        if not html.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    out_path = Path(args.output) if args.output else in_path.with_suffix(".html")
    out_path.write_text(html + ("\n" if not html.endswith("\n") else ""), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
