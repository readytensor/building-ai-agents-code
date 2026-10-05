"""End-to-end fixture-based renderer tests.

For every `tests/fixtures/foo.md`, compare `render(foo.md)` to the contents
of `tests/fixtures/foo.html`. Adding a new test is as cheap as dropping in
a new file pair.
"""

from __future__ import annotations

from pathlib import Path

from md2html import render


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Allow a single trailing newline on the expected HTML — editors add it.
    if text.endswith("\n"):
        text = text[:-1]
    return text


def test_fixture_pair(fixture_pair):
    md_path, html_path = fixture_pair
    md = md_path.read_text(encoding="utf-8")
    expected = _read(html_path)
    actual = render(md)
    assert actual == expected, (
        f"\n--- {md_path.name} ---\n"
        f"Expected:\n{expected!r}\n"
        f"Actual:\n{actual!r}\n"
    )


def test_render_empty():
    assert render("") == ""


def test_render_single_paragraph():
    assert render("hello") == "<p>hello</p>"


def test_render_no_extensions():
    md = "```python\nx = 1\n```"
    # Without code_blocks extension, no language class.
    assert render(md, extensions=[]) == "<pre><code>x = 1</code></pre>"
