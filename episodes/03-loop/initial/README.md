# md2html

A small Markdown-to-HTML CLI tool with a clean three-stage pipeline:

```
markdown text → lexer → tokens → parser → AST → renderer → HTML
```

It implements a deliberately small but real subset of Markdown: ATX headings, paragraphs, ordered/unordered (and nested) lists, fenced code blocks, blockquotes, horizontal rules, and the usual inline constructs (emphasis, strong, inline code, links, images, hard breaks).

Three optional extensions ship in-tree:

- **tables**: GitHub-flavored pipe tables with column alignment.
- **code_blocks**: adds the `language-xxx` class on fenced code blocks.
- **footnotes**: `[^1]` references plus collected definitions at the document end.

## Install

```
pip install -e ".[test]"
```

## Use

```
md2html INPUT_FILE [-o OUTPUT_FILE] [--stdout] [-s | --standalone]
        [--no-extensions] [--extensions LIST]
```

Examples:

```
md2html README.md                       # writes README.html (HTML body fragment)
md2html README.md --standalone          # writes a complete standalone HTML page
md2html README.md --stdout              # prints to stdout
md2html post.md --extensions tables     # tables only
md2html post.md --no-extensions         # core markdown only
```

By default `md2html` emits an HTML body fragment (the usual contract for a Markdown converter); `--standalone` (`-s`) wraps it in a complete HTML document with a built-in stylesheet, ready to open in a browser.

Or invoke as a module:

```
python -m md2html README.md --stdout
```

## Library use

```python
from md2html import render

html = render(open("README.md").read())
```

## Architecture

| Module | Responsibility |
|---|---|
| `md2html/lexer.py` | Scan markdown text into a flat stream of block-level tokens. |
| `md2html/parser.py` | Build an AST from the token stream; do inline parsing. |
| `md2html/renderer.py` | Visitor-pattern walker producing HTML. |
| `md2html/extensions/` | Each extension is one file contributing to all three stages. |
| `md2html/utils.py` | HTML-escape, slugify, whitespace helpers. |
| `md2html/cli.py` | argparse entry point. |

## Tests

```
pytest
```

Tests cover the lexer, the parser, and end-to-end fixture pairs under `tests/fixtures/` (one `*.md` input and one `*.html` expected output per case).

## Scope

This is intentionally a teaching codebase, not a production Markdown engine. The following are **out of scope** by design: HTML pass-through, setext headings, indented code blocks, reference-style links, strikethrough, task lists, autolinks, and math.
