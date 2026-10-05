"""Markdown text -> linear stream of block-level Token objects.

The lexer is intentionally line-oriented: it walks lines one at a time and
emits one token per block construct. Inline content inside a block is held
as raw text and parsed later by `parser.py`.

Indentation is preserved on list items and blockquote lines so the parser
can reconstruct nesting structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .utils import expand_tabs

# ---------------------------------------------------------------------------
# Token type
# ---------------------------------------------------------------------------


@dataclass
class Token:
    """One block-level token.

    `kind` is a string discriminator. `value` carries the raw text payload
    (heading text, paragraph text, code-block body, etc.). `attrs` is a
    free-form bag of extra data — used by list items for indentation/marker,
    by tables for cells/alignment, etc.
    """

    kind: str
    value: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)


# Token kinds the core lexer emits.
TK_HEADING = "heading"
TK_PARAGRAPH = "paragraph"
TK_LIST_ITEM = "list_item"
TK_CODE_BLOCK = "code_block"
TK_BLOCKQUOTE_LINE = "blockquote_line"
TK_HR = "hr"
TK_BLANK = "blank"


# ---------------------------------------------------------------------------
# Patterns (compiled once)
# ---------------------------------------------------------------------------

_RE_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_RE_FENCE = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})[ \t]*(\S*)[ \t]*$")
_RE_HR = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_RE_UL_ITEM = re.compile(r"^([ \t]*)([-*+])[ \t]+(.*)$")
_RE_OL_ITEM = re.compile(r"^([ \t]*)(\d{1,9})([.)])[ \t]+(.*)$")
_RE_BLOCKQUOTE = re.compile(r"^([ \t]{0,3})>[ \t]?(.*)$")
_RE_BLANK = re.compile(r"^[ \t]*$")


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


class Lexer:
    """Block-level scanner.

    Usage:

        tokens = Lexer(markdown_source, extensions=[...]).tokenize()

    Extensions whose objects expose a `tokenize_block(lexer) -> bool` hook
    get a chance to consume input before the built-in rules fire. The hook
    inspects `self.line()` (current line) and may advance via `self.advance()`
    or `self.emit()`.
    """

    def __init__(self, source: str, extensions: list | None = None) -> None:
        # Normalise line endings + tab expansion up front so downstream code
        # only ever sees \n and spaces.
        source = expand_tabs(source.replace("\r\n", "\n").replace("\r", "\n"))
        self.lines: list[str] = source.split("\n")
        self.pos: int = 0
        self.tokens: list[Token] = []
        self.extensions = extensions or []

    # -- cursor helpers -----------------------------------------------------

    def eof(self) -> bool:
        return self.pos >= len(self.lines)

    def line(self) -> str:
        return self.lines[self.pos] if not self.eof() else ""

    def peek(self, offset: int = 1) -> str:
        idx = self.pos + offset
        return self.lines[idx] if 0 <= idx < len(self.lines) else ""

    def advance(self, n: int = 1) -> None:
        self.pos += n

    def emit(self, token: Token) -> None:
        self.tokens.append(token)

    # -- main loop ----------------------------------------------------------

    def tokenize(self) -> list[Token]:
        while not self.eof():
            line = self.line()

            if _RE_BLANK.match(line):
                self.emit(Token(TK_BLANK))
                self.advance()
                continue

            # Give extensions first crack at the current line.
            if self._try_extensions():
                continue

            if _RE_HR.match(line):
                self.emit(Token(TK_HR))
                self.advance()
                continue

            m = _RE_HEADING.match(line)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                self.emit(Token(TK_HEADING, text, {"level": level}))
                self.advance()
                continue

            if _RE_FENCE.match(line):
                self._consume_fenced_code()
                continue

            if _RE_BLOCKQUOTE.match(line):
                self._consume_blockquote()
                continue

            if _RE_UL_ITEM.match(line) or _RE_OL_ITEM.match(line):
                self._consume_list_item()
                continue

            # Default: paragraph (consume until blank/other block starter).
            self._consume_paragraph()

        return self.tokens

    # -- extension hook -----------------------------------------------------

    def _try_extensions(self) -> bool:
        for ext in self.extensions:
            hook = getattr(ext, "tokenize_block", None)
            if hook is None:
                continue
            before = self.pos
            if hook(self):
                # Hook must have advanced; otherwise we'd loop forever.
                if self.pos == before:
                    raise RuntimeError(
                        f"Extension {ext.name!r} returned True without advancing"
                    )
                return True
        return False

    # -- block consumers ----------------------------------------------------

    def _consume_fenced_code(self) -> None:
        """Consume an opening fence, body, and closing fence."""
        m = _RE_FENCE.match(self.line())
        assert m is not None
        fence = m.group(2)
        info = m.group(3)
        self.advance()

        body_lines: list[str] = []
        while not self.eof():
            line = self.line()
            mc = _RE_FENCE.match(line)
            # Closing fence: same char, at least as long, no info string.
            if mc and mc.group(2)[0] == fence[0] and len(mc.group(2)) >= len(fence) and not mc.group(3):
                self.advance()
                break
            body_lines.append(line)
            self.advance()

        body = "\n".join(body_lines)
        self.emit(Token(TK_CODE_BLOCK, body, {"info": info, "lang": info or ""}))

    def _consume_blockquote(self) -> None:
        """Consume one or more consecutive blockquote lines into a single
        token (parser will reparse the content).
        """
        collected: list[str] = []
        while not self.eof():
            m = _RE_BLOCKQUOTE.match(self.line())
            if not m:
                break
            collected.append(m.group(2))
            self.advance()
        self.emit(Token(TK_BLOCKQUOTE_LINE, "\n".join(collected)))

    def _consume_list_item(self) -> None:
        """Consume one list item (one marker line plus any continuation
        lines). The parser glues sibling items into a list and handles
        nesting via the indent attribute.
        """
        line = self.line()
        m_ul = _RE_UL_ITEM.match(line)
        m_ol = _RE_OL_ITEM.match(line)

        if m_ul:
            indent = len(m_ul.group(1))
            marker = m_ul.group(2)
            content = m_ul.group(3)
            ordered = False
            start = None
        else:
            assert m_ol is not None
            indent = len(m_ol.group(1))
            marker = m_ol.group(2) + m_ol.group(3)
            content = m_ol.group(4)
            ordered = True
            start = int(m_ol.group(2))

        # The "content indent" is where continuation lines must start to be
        # considered part of this item. It's the column right after the
        # marker + 1 space.
        marker_len = (len(m_ul.group(2)) if m_ul else len(m_ol.group(2)) + 1)
        content_indent = indent + marker_len + 1

        self.advance()
        body_lines = [content]

        while not self.eof():
            nxt = self.line()
            if _RE_BLANK.match(nxt):
                # Blank line may continue the item if a sufficiently indented
                # line follows; otherwise it terminates.
                # Tight-lists semantics (per spec): we treat blank as
                # terminator unless followed by indented content.
                la = self.peek()
                if not la or _RE_BLANK.match(la):
                    break
                la_indent = len(la) - len(la.lstrip(" "))
                if la_indent < content_indent and not (
                    _RE_UL_ITEM.match(la) or _RE_OL_ITEM.match(la)
                ):
                    break
                body_lines.append("")
                self.advance()
                continue

            # Another list item at any indent level breaks the current item.
            if _RE_UL_ITEM.match(nxt) or _RE_OL_ITEM.match(nxt):
                break

            # Continuation of the current item: must be indented enough.
            cur_indent = len(nxt) - len(nxt.lstrip(" "))
            if cur_indent >= content_indent:
                body_lines.append(nxt[content_indent:])
                self.advance()
                continue

            # Lazy continuation: a non-blank, non-block-starter, less-indented
            # line still belongs to the paragraph inside this item.
            if (
                cur_indent < content_indent
                and not _RE_BLOCKQUOTE.match(nxt)
                and not _RE_HR.match(nxt)
                and not _RE_HEADING.match(nxt)
                and not _RE_FENCE.match(nxt)
            ):
                body_lines.append(nxt.lstrip(" "))
                self.advance()
                continue

            break

        body = "\n".join(body_lines).rstrip("\n")
        self.emit(
            Token(
                TK_LIST_ITEM,
                body,
                {
                    "indent": indent,
                    "ordered": ordered,
                    "marker": marker,
                    "start": start,
                },
            )
        )

    def _consume_paragraph(self) -> None:
        """Consume one paragraph: lines until blank or a non-paragraph block
        boundary.
        """
        lines = [self.line()]
        self.advance()
        while not self.eof():
            nxt = self.line()
            if _RE_BLANK.match(nxt):
                break
            if (
                _RE_HEADING.match(nxt)
                or _RE_HR.match(nxt)
                or _RE_FENCE.match(nxt)
                or _RE_BLOCKQUOTE.match(nxt)
                or _RE_UL_ITEM.match(nxt)
                or _RE_OL_ITEM.match(nxt)
            ):
                break
            # Let extensions break paragraph continuation too (e.g. tables).
            if self._extension_breaks_paragraph(nxt):
                break
            lines.append(nxt)
            self.advance()
        # Join with single spaces collapsed at newline boundaries (but keep
        # hard-break double-space markers for the inline pass).
        text = "\n".join(lines)
        self.emit(Token(TK_PARAGRAPH, text))

    def _extension_breaks_paragraph(self, line: str) -> bool:
        for ext in self.extensions:
            hook = getattr(ext, "breaks_paragraph", None)
            if hook is not None and hook(line):
                return True
        return False
