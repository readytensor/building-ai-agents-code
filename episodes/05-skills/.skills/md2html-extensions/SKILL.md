---
name: md2html-extensions
description: How md2html is extended, its hooks, the registry, and the way extensions are tested. Use when a task adds a feature to md2html, before reading the code.
---

# Extending md2html

md2html is a three-stage pipeline, lexer to parser to renderer, and every
extension is a small class in `md2html/extensions/<name>.py` with a `name`
attribute and any of these hooks. Each stage checks for the hook by name and
gives extensions the first chance at every step.

| Stage | Hook | Returns |
|---|---|---|
| lexer | `tokenize_block(self, lexer)` | `True` after emitting a token, advanced |
| lexer | `breaks_paragraph(self, line)` | `True` if this line ends a paragraph |
| parser | `parse_block(self, parser, tok)` | a `Node`, or `None` to pass |
| parser | `parse_inline(self, parser, text, i, out, buf)` | characters used, or `0` |
| parser | `post_parse(self, root, parser)` | nothing; edits the tree in place |
| renderer | `render(self, renderer, node)` | an HTML string, or `None` to pass |
| renderer | `post_render(self, renderer, html, root)` | the final HTML |

Nodes are `Node(kind, value=..., attrs=...)` from `..parser`. An inline hook
that emits a node first flushes the pending text: `out.append(Node("text",
value="".join(buf))); buf.clear()`. Code spans and fenced code blocks are
consumed by the core before any inline hook runs, so their contents are
protected without extra work. The renderer escapes the text of `text` nodes;
a `render` hook returns finished HTML.

## Registering

Add the class to `_REGISTRY` in `md2html/extensions/__init__.py`. The order is
the order hooks run in: `tables` first (its block token competes with
paragraphs), `footnotes` last (it appends a section after rendering). A
registered extension is on by default and can be named in
`render(text, extensions="tables,footnotes")` or `--extensions`.

## Testing

- `tests/fixtures/<name>.md` with `<name>.html` beside it: `tests/test_renderer.py`
  renders every pair and compares, so a fixture pair is a test with no code.
- Unit tests go in `tests/test_<name>.py`, in the style of `tests/test_parser.py`.
- Run everything with `python -m pytest -q` from the project folder.
