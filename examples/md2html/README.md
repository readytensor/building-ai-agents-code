# md2html

md2html converts Markdown to HTML. It is a small Python package with a
command-line tool, no runtime dependencies, and a three-stage pipeline:

```
Markdown → lexer → tokens → parser → AST → renderer → HTML
```

## Install

md2html needs Python 3.11 or newer. From the project folder:

```sh
pip install -e ".[test]"
```

This installs the `md2html` command, and pytest for the test suite.

## Command line

```sh
md2html notes.md                      # writes notes.html next to the input
md2html notes.md -o page.html         # writes to a file you name
md2html notes.md --stdout             # prints the HTML instead
md2html notes.md --standalone         # a full page with a stylesheet
```

Without `--standalone`, the output is an HTML fragment for the page body.
With it, the fragment is wrapped in a full HTML document; the page title
comes from the first `#` heading, or from the file name if there is none.
`python -m md2html` works the same as the `md2html` command.

## In Python

```python
from md2html import render

render("# Hello\n\nSome *Markdown*.")
# '<h1>Hello</h1>\n<p>Some <em>Markdown</em>.</p>'
```

## What it supports

Headings (`#` to `######`), paragraphs, ordered and unordered lists
(nested by indentation), blockquotes (which can contain other blocks),
fenced code blocks, horizontal rules, and inline emphasis, strong text,
code spans, links, images and hard line breaks.

Eight extensions are on by default:

- **tables**: GitHub-style pipe tables, with column alignment.
- **code_blocks**: a `language-…` class on fenced code blocks.
- **footnotes**: `[^1]` references, with the notes collected at the end.
- **reference_links**: `[text][id]` links, resolved against `[id]: url "title"` lines.
- **github_alerts**: `> [!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]` and
  `[!CAUTION]` blockquotes, rendered as GitHub's alert boxes.
- **strikethrough**: `~~text~~` becomes `<del>text</del>`.
- **task_lists**: `- [ ]` and `- [x]` list items get a checkbox.
- **autolinks**: `<https://example.com>` becomes a link.

Turn them off with `--no-extensions`, or pick some with
`--extensions tables,footnotes` (in Python: `render(text, extensions="tables")`).

Not supported: raw HTML pass-through, setext (underlined) headings, and
indented code blocks.

## How it works

| Module | What it does |
|---|---|
| `md2html/lexer.py` | Splits the text into a flat stream of block-level tokens. |
| `md2html/parser.py` | Builds the tree of nodes (the AST) and parses inline syntax. |
| `md2html/renderer.py` | Walks the tree and writes HTML. |
| `md2html/extensions/` | One file per extension; an extension can hook into any of the three stages. |
| `md2html/cli.py` | The command-line tool. |
| `md2html/utils.py` | Small helpers: HTML escaping, slugs, whitespace. |

## Tests

```sh
pytest
```

There are unit tests for the lexer and the parser, and end-to-end tests
built from pairs of files in `tests/fixtures/`: each `name.md` is rendered
and compared with `name.html`. To add a case, add a new pair.
