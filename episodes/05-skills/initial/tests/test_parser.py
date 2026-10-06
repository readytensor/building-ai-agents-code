"""Parser unit tests.

These tests construct an AST and assert on its shape. Inline parsing is
tested via the renderer fixtures, but a few targeted parser-only tests
exercise the trickier nesting cases (lists in blockquotes, lists in lists,
blockquotes in blockquotes).
"""

from __future__ import annotations

from md2html.extensions import default_extensions
from md2html.lexer import Lexer
from md2html.parser import Node, Parser


def _parse(source: str, extensions=None) -> Node:
    tokens = Lexer(source, extensions=extensions or []).tokenize()
    return Parser(tokens, extensions=extensions or []).parse()


def test_document_root():
    root = _parse("hi\n")
    assert root.kind == "document"
    assert len(root.children) == 1
    assert root.children[0].kind == "paragraph"


def test_heading_carries_level():
    root = _parse("### Title\n")
    h = root.children[0]
    assert h.kind == "heading"
    assert h.attrs["level"] == 3


def test_paragraph_inline_children_kinds():
    root = _parse("a *b* **c** `d`\n")
    p = root.children[0]
    kinds = [c.kind for c in p.children]
    assert "emph" in kinds
    assert "strong" in kinds
    assert "code" in kinds


def test_link_and_image_attrs():
    root = _parse("[t](u) ![a](s)\n")
    p = root.children[0]
    link = next(c for c in p.children if c.kind == "link")
    img = next(c for c in p.children if c.kind == "image")
    assert link.attrs["href"] == "u"
    assert img.attrs["src"] == "s"
    assert img.value == "a"


def test_unordered_list_structure():
    root = _parse("- a\n- b\n")
    lst = root.children[0]
    assert lst.kind == "list"
    assert lst.attrs["ordered"] is False
    assert len(lst.children) == 2
    assert all(item.kind == "list_item" for item in lst.children)


def test_ordered_list_start():
    root = _parse("3. three\n4. four\n")
    lst = root.children[0]
    assert lst.attrs["ordered"] is True
    assert lst.attrs["start"] == 3


def test_nested_list_inside_item():
    root = _parse("- outer\n  - inner\n")
    outer_list = root.children[0]
    outer_item = outer_list.children[0]
    # Outer item should contain inline text + a nested list.
    nested = [c for c in outer_item.children if c.kind == "list"]
    assert nested, "expected nested list inside item"
    assert nested[0].children[0].kind == "list_item"


def test_blockquote_contains_paragraph():
    root = _parse("> hello\n")
    bq = root.children[0]
    assert bq.kind == "blockquote"
    assert bq.children[0].kind == "paragraph"


def test_blockquote_contains_list():
    root = _parse("> - a\n> - b\n")
    bq = root.children[0]
    assert bq.kind == "blockquote"
    assert any(c.kind == "list" for c in bq.children)


def test_blockquote_in_blockquote():
    root = _parse("> > deep\n")
    outer = root.children[0]
    assert outer.kind == "blockquote"
    inner = outer.children[0]
    assert inner.kind == "blockquote"


def test_horizontal_rule_node():
    root = _parse("---\n")
    assert root.children[0].kind == "hr"


def test_code_block_value_preserved():
    root = _parse("```\nline1\nline2\n```\n")
    cb = root.children[0]
    assert cb.kind == "code_block"
    assert cb.value == "line1\nline2"


def test_table_node_shape():
    root = _parse(
        "| a | b |\n|---|---|\n| 1 | 2 |\n",
        extensions=default_extensions(),
    )
    table = root.children[0]
    assert table.kind == "table"
    # First child is the header row, rest are body rows.
    assert table.children[0].kind == "table_header"
    assert table.children[1].kind == "table_row"


def test_footnote_def_extracted_from_tree():
    root = _parse(
        "Hi[^1].\n\n[^1]: there\n",
        extensions=default_extensions(),
    )
    # The footnote_def should not appear as a top-level child; it's stashed
    # in root.attrs.
    kinds = [c.kind for c in root.children]
    assert "footnote_def" not in kinds
    assert "footnote_defs" in root.attrs
    assert "1" in root.attrs["footnote_defs"]
