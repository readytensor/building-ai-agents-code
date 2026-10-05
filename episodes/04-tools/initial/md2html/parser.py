"""Token stream -> AST.

The parser handles two responsibilities:

1. **Block nesting.** The lexer is flat; the parser groups list items into
   lists, recursively parses blockquote bodies, and resolves nested-list
   indentation.

2. **Inline parsing.** Within a block's raw text, the parser walks the
   string once to produce inline nodes for emphasis, strong, code spans,
   links, images, and hard breaks.

Inline parsing is a deliberately small hand-rolled scanner — not a regex
soup — so the recursion for emphasis-inside-link-text-inside-emphasis stays
legible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .lexer import (
    TK_BLANK,
    TK_BLOCKQUOTE_LINE,
    TK_CODE_BLOCK,
    TK_HEADING,
    TK_HR,
    TK_LIST_ITEM,
    TK_PARAGRAPH,
    Lexer,
    Token,
)


# ---------------------------------------------------------------------------
# AST node
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """An AST node.

    `kind` is the discriminator the renderer dispatches on (e.g. "heading",
    "paragraph", "text", "emph"). `children` carries child nodes. `value`
    holds a string payload for leaf nodes (text, code spans, code blocks).
    `attrs` is a free-form bag for kind-specific metadata.
    """

    kind: str
    children: list["Node"] = field(default_factory=list)
    value: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    def __init__(self, tokens: list[Token], extensions: list | None = None) -> None:
        self.tokens = tokens
        self.pos = 0
        self.extensions = extensions or []

    # -- cursor helpers -----------------------------------------------------

    def eof(self) -> bool:
        return self.pos >= len(self.tokens)

    def current(self) -> Token | None:
        return self.tokens[self.pos] if not self.eof() else None

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    # -- public API ---------------------------------------------------------

    def parse(self) -> Node:
        root = Node("document")
        while not self.eof():
            node = self._parse_block()
            if node is not None:
                root.children.append(node)
        # Let extensions do post-parse passes (footnotes pulls defs out).
        for ext in self.extensions:
            hook = getattr(ext, "post_parse", None)
            if hook is not None:
                hook(root, self)
        return root

    # -- block dispatch -----------------------------------------------------

    def _parse_block(self) -> Node | None:
        tok = self.current()
        assert tok is not None

        if tok.kind == TK_BLANK:
            self.advance()
            return None

        # Extension AST hook: a non-core token kind may have been emitted by
        # an extension lexer hook. We let the extension claim it here.
        for ext in self.extensions:
            hook = getattr(ext, "parse_block", None)
            if hook is not None:
                node = hook(self, tok)
                if node is not None:
                    return node

        if tok.kind == TK_HEADING:
            self.advance()
            return Node(
                "heading",
                children=self._parse_inline(tok.value),
                attrs={"level": tok.attrs["level"]},
            )

        if tok.kind == TK_PARAGRAPH:
            self.advance()
            return Node("paragraph", children=self._parse_inline(tok.value))

        if tok.kind == TK_HR:
            self.advance()
            return Node("hr")

        if tok.kind == TK_CODE_BLOCK:
            self.advance()
            return Node(
                "code_block",
                value=tok.value,
                attrs={"lang": tok.attrs.get("lang", ""), "info": tok.attrs.get("info", "")},
            )

        if tok.kind == TK_BLOCKQUOTE_LINE:
            self.advance()
            # Re-lex the blockquote body so any block construct inside it
            # (lists, code, nested blockquotes) parses cleanly.
            inner_tokens = Lexer(tok.value, extensions=self.extensions).tokenize()
            inner = Parser(inner_tokens, extensions=self.extensions).parse()
            inner.kind = "blockquote"
            return inner

        if tok.kind == TK_LIST_ITEM:
            return self._parse_list()

        # Unknown token kind — skip with a graceful no-op rather than crash.
        self.advance()
        return None

    # -- list parsing -------------------------------------------------------

    def _parse_list(self) -> Node:
        first = self.current()
        assert first is not None and first.kind == TK_LIST_ITEM
        ordered = first.attrs["ordered"]
        base_indent = first.attrs["indent"]
        start = first.attrs.get("start")

        list_node = Node(
            "list",
            attrs={"ordered": ordered, "start": start if ordered and start != 1 else None},
        )

        while not self.eof():
            tok = self.current()
            if tok is None:
                break
            if tok.kind == TK_BLANK:
                # Peek past blanks: if the next non-blank is another item at
                # this list's level, keep going; otherwise stop.
                save = self.pos
                while not self.eof() and self.tokens[self.pos].kind == TK_BLANK:
                    self.pos += 1
                if self.eof():
                    break
                nxt = self.tokens[self.pos]
                if (
                    nxt.kind != TK_LIST_ITEM
                    or nxt.attrs["indent"] < base_indent
                    or (nxt.attrs["indent"] == base_indent and nxt.attrs["ordered"] != ordered)
                ):
                    # Rewind so the caller still sees the blank tokens.
                    self.pos = save
                    break
                # Skip the blanks and continue with the next item.
                continue
            if tok.kind != TK_LIST_ITEM:
                break
            if tok.attrs["indent"] < base_indent:
                break
            if tok.attrs["indent"] == base_indent and tok.attrs["ordered"] != ordered:
                break

            if tok.attrs["indent"] > base_indent:
                # This item is more deeply indented than our list's base —
                # it belongs as a nested sub-list inside the last item.
                if not list_node.children:
                    # No item yet to nest under; treat as sibling.
                    self.advance()
                    list_node.children.append(
                        Node("list_item", children=self._parse_item_body(tok))
                    )
                else:
                    sublist = self._parse_list()
                    list_node.children[-1].children.append(sublist)
                continue

            self.advance()
            item_node = Node("list_item", children=self._parse_item_body(tok))
            list_node.children.append(item_node)

        return list_node

    def _parse_item_body(self, tok: Token) -> list[Node]:
        """Parse a single list item's body: the marker-line text plus any
        continuation lines, possibly containing nested lists or blocks.

        We re-lex the body text. The body has already been dedented to the
        item's content column by the lexer.
        """
        body = tok.value
        # Detect nested-list indentation. A nested list is any line beginning
        # with a list marker at indent > 0.
        inner_tokens = Lexer(body, extensions=self.extensions).tokenize()
        inner = Parser(inner_tokens, extensions=self.extensions).parse()

        # If the body produced exactly one paragraph, unwrap it so we get
        # "tight" rendering (<li>foo</li> rather than <li><p>foo</p></li>).
        children = inner.children
        if len(children) == 1 and children[0].kind == "paragraph":
            return children[0].children
        if (
            len(children) >= 1
            and children[0].kind == "paragraph"
            and all(c.kind in ("list", "paragraph") for c in children)
        ):
            # First paragraph is inline (tight), then nested blocks.
            head = children[0].children
            tail = children[1:]
            return head + tail
        return children

    # -- inline parsing -----------------------------------------------------

    _INLINE_CHARS = set("*_`[!\\\n")

    def _parse_inline(self, text: str) -> list[Node]:
        """Walk `text` once and emit a flat list of inline nodes."""
        out: list[Node] = []
        i = 0
        n = len(text)
        buf: list[str] = []

        def flush() -> None:
            if buf:
                out.append(Node("text", value="".join(buf)))
                buf.clear()

        while i < n:
            ch = text[i]

            # Let extensions claim inline syntax first (e.g., footnotes
            # consume the [^x] form before the link parser sees it).
            consumed = self._try_inline_extensions(text, i, out, buf)
            if consumed > 0:
                i += consumed
                continue

            if ch == "\\" and i + 1 < n and text[i + 1] in "\\`*_{}[]()#+-.!|>":
                # Backslash escape.
                buf.append(text[i + 1])
                i += 2
                continue

            if ch == "\n":
                # Hard break if previous chars are "  " (two trailing spaces).
                if len(buf) >= 2 and buf[-1] == " " and buf[-2] == " ":
                    buf.pop()
                    buf.pop()
                    flush()
                    out.append(Node("linebreak"))
                else:
                    # Soft break -> single space.
                    buf.append(" ")
                i += 1
                # Collapse leading whitespace on the next line.
                while i < n and text[i] == " ":
                    i += 1
                continue

            if ch == "`":
                end, code = self._match_code_span(text, i)
                if end > 0:
                    flush()
                    out.append(Node("code", value=code))
                    i = end
                    continue

            if ch == "!" and i + 1 < n and text[i + 1] == "[":
                end, alt, url, title = self._match_link(text, i + 1)
                if end > 0:
                    flush()
                    img = Node("image", value=alt, attrs={"src": url, "title": title})
                    out.append(img)
                    i = end
                    continue

            if ch == "[":
                end, label, url, title = self._match_link(text, i)
                if end > 0:
                    flush()
                    link = Node(
                        "link",
                        children=self._parse_inline(label),
                        attrs={"href": url, "title": title},
                    )
                    out.append(link)
                    i = end
                    continue

            if ch in "*_":
                end, kind, inner = self._match_emphasis(text, i)
                if end > 0:
                    flush()
                    out.append(Node(kind, children=self._parse_inline(inner)))
                    i = end
                    continue

            buf.append(ch)
            i += 1

        flush()
        return out

    def _try_inline_extensions(
        self, text: str, i: int, out: list[Node], buf: list[str]
    ) -> int:
        """Give extensions first crack at inline parsing. Returns the number
        of input characters consumed, or 0 if no extension claimed it.

        Extensions that want to emit a node should append it to `out` after
        flushing `buf` themselves (we provide a helper here).
        """
        for ext in self.extensions:
            hook = getattr(ext, "parse_inline", None)
            if hook is None:
                continue
            consumed = hook(self, text, i, out, buf)
            if consumed and consumed > 0:
                return consumed
        return 0

    # -- inline helpers -----------------------------------------------------

    @staticmethod
    def _match_code_span(text: str, start: int) -> tuple[int, str]:
        """Match an inline code span starting at a backtick. Returns
        (end_index, code) or (0, "") if no match.

        Supports 1+ backticks as fence; the closing fence must have the
        same length.
        """
        n = len(text)
        j = start
        while j < n and text[j] == "`":
            j += 1
        fence_len = j - start
        # Find closing fence of the same length.
        k = j
        while k < n:
            if text[k] == "`":
                p = k
                while p < n and text[p] == "`":
                    p += 1
                if p - k == fence_len:
                    code = text[j:k]
                    # CommonMark-ish: strip one leading and trailing space if
                    # both ends have space and the content isn't entirely
                    # whitespace.
                    if (
                        len(code) >= 2
                        and code.startswith(" ")
                        and code.endswith(" ")
                        and code.strip() != ""
                    ):
                        code = code[1:-1]
                    return p, code
                k = p
            else:
                k += 1
        return 0, ""

    _LINK_INLINE_RE = re.compile(
        r"""\(
            \s*
            (?P<url>(?:<[^>]*>|[^\s)]*))     # url, possibly <bracketed>
            (?:
                \s+
                (?P<title>"[^"]*"|'[^']*'|\([^)]*\))
            )?
            \s*
            \)""",
        re.VERBOSE,
    )

    def _match_link(self, text: str, start: int) -> tuple[int, str, str, str]:
        """Match `[label](url "title")` starting at `[`. Returns
        (end_index, label, url, title) or (0, "", "", "") if no match.
        """
        assert text[start] == "["
        # Find matching `]` accounting for nested brackets in label.
        depth = 1
        i = start + 1
        n = len(text)
        while i < n and depth > 0:
            c = text[i]
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0 or i >= n:
            return 0, "", "", ""
        label = text[start + 1 : i]
        # Now expect `(`...
        if i + 1 >= n or text[i + 1] != "(":
            return 0, "", "", ""
        m = self._LINK_INLINE_RE.match(text, i + 1)
        if not m:
            return 0, "", "", ""
        url = m.group("url") or ""
        if url.startswith("<") and url.endswith(">"):
            url = url[1:-1]
        title = m.group("title") or ""
        if title and title[0] in "\"'(":
            title = title[1:-1]
        return m.end(), label, url, title

    def _match_emphasis(self, text: str, start: int) -> tuple[int, str, str]:
        """Match emphasis or strong starting at `*` or `_`. Returns
        (end_index, kind, inner) where kind is "emph" or "strong".

        Simple, non-CommonMark-exhaustive: we look for the next matching
        same-character run of the same length.
        """
        n = len(text)
        ch = text[start]
        # Count opening run.
        j = start
        while j < n and text[j] == ch and (j - start) < 3:
            j += 1
        run = j - start
        if run == 0:
            return 0, "", ""

        # Opening run can't be followed by whitespace.
        if j >= n or text[j].isspace():
            return 0, "", ""

        # `_` emphasis: respect word boundaries on the open side. If the
        # preceding char is alphanumeric (e.g. "snake_case"), bail.
        if ch == "_" and start > 0 and text[start - 1].isalnum():
            return 0, "", ""

        # Search for matching closing run.
        k = j
        while k < n:
            if text[k] == "\\" and k + 1 < n:
                k += 2
                continue
            if text[k] == ch:
                p = k
                while p < n and text[p] == ch:
                    p += 1
                close_run = p - k
                if close_run >= run and not text[k - 1].isspace():
                    # `_` close needs to not be followed by alnum.
                    if ch == "_" and p < n and text[p].isalnum():
                        k = p
                        continue
                    # Use `run` chars of the close fence.
                    inner = text[j:k]
                    end = k + run
                    kind = "strong" if run >= 2 else "emph"
                    if run == 3:
                        # ***x*** -> <strong><em>x</em></strong>.
                        return end, "emph_strong", inner
                    return end, kind, inner
                k = p
            else:
                k += 1
        return 0, "", ""
