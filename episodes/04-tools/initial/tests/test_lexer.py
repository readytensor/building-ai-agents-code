"""Lexer unit tests.

These tests check the raw token stream produced for each block construct.
They guard the public contract between lexer and parser: parsers should
trust the token kinds and attrs the lexer emits.
"""

from __future__ import annotations

from md2html.extensions import default_extensions
from md2html.lexer import (
    TK_BLANK,
    TK_BLOCKQUOTE_LINE,
    TK_CODE_BLOCK,
    TK_HEADING,
    TK_HR,
    TK_LIST_ITEM,
    TK_PARAGRAPH,
    Lexer,
)


def _kinds(source: str, extensions=None) -> list[str]:
    return [t.kind for t in Lexer(source, extensions=extensions or []).tokenize()]


def test_empty_input():
    tokens = Lexer("").tokenize()
    assert [t.kind for t in tokens] == [TK_BLANK]


def test_single_paragraph():
    tokens = Lexer("hello world").tokenize()
    assert tokens[0].kind == TK_PARAGRAPH
    assert tokens[0].value == "hello world"


def test_atx_heading_levels():
    src = "# h1\n## h2\n###### h6\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert [t.kind for t in tokens] == [TK_HEADING] * 3
    assert [t.attrs["level"] for t in tokens] == [1, 2, 6]
    assert [t.value for t in tokens] == ["h1", "h2", "h6"]


def test_seven_hashes_is_paragraph():
    # 7+ hashes is not a heading.
    tokens = Lexer("####### too many").tokenize()
    assert tokens[0].kind == TK_PARAGRAPH


def test_horizontal_rule_variants():
    for line in ("---", "***", "___", "- - -", "  ---"):
        tokens = Lexer(line + "\n").tokenize()
        assert tokens[0].kind == TK_HR, f"expected HR for {line!r}"


def test_blank_lines_emit_blank_tokens():
    tokens = Lexer("a\n\n\nb\n").tokenize()
    kinds = [t.kind for t in tokens]
    assert TK_BLANK in kinds


def test_fenced_code_captures_lang_and_body():
    src = "```python\nx = 1\ny = 2\n```\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert tokens[0].kind == TK_CODE_BLOCK
    assert tokens[0].value == "x = 1\ny = 2"
    assert tokens[0].attrs["lang"] == "python"


def test_fenced_code_no_lang():
    src = "```\nplain\n```\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert tokens[0].kind == TK_CODE_BLOCK
    assert tokens[0].attrs["lang"] == ""


def test_blockquote_collects_consecutive_lines():
    src = "> one\n> two\n> three\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert tokens[0].kind == TK_BLOCKQUOTE_LINE
    assert tokens[0].value == "one\ntwo\nthree"


def test_unordered_list_items():
    src = "- a\n- b\n* c\n+ d\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert all(t.kind == TK_LIST_ITEM for t in tokens)
    assert [t.attrs["ordered"] for t in tokens] == [False] * 4


def test_ordered_list_items():
    src = "1. first\n2. second\n10. tenth\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert all(t.kind == TK_LIST_ITEM for t in tokens)
    assert all(t.attrs["ordered"] for t in tokens)
    assert [t.attrs["start"] for t in tokens] == [1, 2, 10]


def test_nested_list_indent_preserved():
    src = "- outer\n  - inner\n- next\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert [t.attrs["indent"] for t in tokens] == [0, 2, 0]


def test_tab_normalization():
    # A tab at the start of a line becomes 4 spaces, so this is a nested
    # list item under "outer" (indent=4).
    src = "- outer\n\t- inner\n"
    tokens = [t for t in Lexer(src).tokenize() if t.kind != TK_BLANK]
    assert tokens[1].attrs["indent"] == 4


def test_table_extension_emits_table_token():
    src = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    tokens = [t for t in Lexer(src, extensions=default_extensions()).tokenize() if t.kind != TK_BLANK]
    assert tokens[0].kind == "table"
    assert tokens[0].attrs["header"] == ["a", "b"]
    assert tokens[0].attrs["rows"] == [["1", "2"]]


def test_footnote_extension_def_token():
    src = "[^1]: a footnote\n"
    tokens = [t for t in Lexer(src, extensions=default_extensions()).tokenize() if t.kind != TK_BLANK]
    assert tokens[0].kind == "footnote_def"
    assert tokens[0].attrs["key"] == "1"
