"""Footnote extension.

Syntax:

    Here is a footnote[^1].

    [^1]: This is the footnote text.

Inline `[^key]` becomes `<sup><a href="#fn-key">N</a></sup>` where N is the
1-based order of first appearance. Definitions are collected and appended
as a `<section class="footnotes">` at the very end of the document.

Definitions can appear anywhere; numeric or named keys both work.
"""

from __future__ import annotations

import re

from ..lexer import Token
from ..parser import Node

TK_FOOTNOTE_DEF = "footnote_def"

# `[^key]: rest of the paragraph...`
_RE_FOOTNOTE_DEF = re.compile(r"^[ \t]{0,3}\[\^([^\]]+)\]:[ \t]*(.*)$")
# Inline ref `[^key]`. Key may be alphanumeric, underscore, dash.
_RE_FOOTNOTE_REF = re.compile(r"\[\^([^\]]+)\]")


class FootnotesExtension:
    name = "footnotes"

    # ------------------------------------------------------------------
    # Lexer hook: capture definition blocks.
    # ------------------------------------------------------------------

    def tokenize_block(self, lexer) -> bool:
        line = lexer.line()
        m = _RE_FOOTNOTE_DEF.match(line)
        if not m:
            return False
        key = m.group(1).strip()
        body_lines = [m.group(2)]
        lexer.advance()
        # Continuation lines: any indented (>= 2 spaces) non-blank line.
        while not lexer.eof():
            nxt = lexer.line()
            if not nxt.strip():
                break
            if _RE_FOOTNOTE_DEF.match(nxt):
                break
            stripped = nxt.lstrip(" ")
            if len(nxt) - len(stripped) >= 2:
                body_lines.append(stripped)
                lexer.advance()
                continue
            break
        body = "\n".join(body_lines).strip()
        lexer.emit(Token(TK_FOOTNOTE_DEF, body, {"key": key}))
        return True

    # ------------------------------------------------------------------
    # Parser hooks: inline refs + def collection.
    # ------------------------------------------------------------------

    def parse_inline(self, parser, text, i, out, buf):
        if text[i] != "[":
            return 0
        if i + 1 >= len(text) or text[i + 1] != "^":
            return 0
        m = _RE_FOOTNOTE_REF.match(text, i)
        if not m:
            return 0
        key = m.group(1).strip()
        if not key:
            return 0
        # Flush the running text buffer before emitting our node.
        if buf:
            out.append(Node("text", value="".join(buf)))
            buf.clear()
        out.append(Node("footnote_ref", attrs={"key": key}))
        return m.end() - i

    def parse_block(self, parser, tok):
        if tok.kind != TK_FOOTNOTE_DEF:
            return None
        parser.advance()
        # Inline-parse the body so links/emphasis inside footnote text work.
        children = parser._parse_inline(tok.value)
        return Node("footnote_def", children=children, attrs={"key": tok.attrs["key"]})

    def post_parse(self, root, parser):
        """Pull all footnote_def nodes out of the tree and stash them on the
        root for the renderer to emit at the end.
        """
        defs: dict[str, Node] = {}
        new_children: list[Node] = []
        for child in root.children:
            self._extract_defs(child, defs)
            if child.kind == "footnote_def":
                defs.setdefault(child.attrs["key"], child)
            else:
                new_children.append(child)
        root.children = new_children
        root.attrs["footnote_defs"] = defs

    def _extract_defs(self, node, defs):
        """Recursively pull footnote_def nodes out of nested containers
        (lists, blockquotes). Mutates `node.children`.
        """
        if not node.children:
            return
        kept: list[Node] = []
        for c in node.children:
            if c.kind == "footnote_def":
                defs.setdefault(c.attrs["key"], c)
            else:
                self._extract_defs(c, defs)
                kept.append(c)
        node.children = kept

    # ------------------------------------------------------------------
    # Renderer hooks.
    # ------------------------------------------------------------------

    def init_state(self, renderer):
        # Maps key -> 1-based number assigned on first reference.
        renderer.state["footnote_order"] = {}

    def render(self, renderer, node):
        if node.kind == "footnote_ref":
            order = renderer.state["footnote_order"]
            key = node.attrs["key"]
            if key not in order:
                order[key] = len(order) + 1
            n = order[key]
            return f'<sup><a href="#fn-{key}">{n}</a></sup>'
        if node.kind == "footnote_def":
            # Suppressed inline — emitted at end of document via post_render.
            return ""
        return None

    def post_render(self, renderer, html, doc_root):
        order = renderer.state.get("footnote_order", {})
        defs = doc_root.attrs.get("footnote_defs", {})
        # Only emit a section for refs we actually saw, in reference order,
        # but include unreferenced definitions at the end too.
        if not order and not defs:
            return html
        items: list[str] = []
        seen_keys: set[str] = set()
        # Referenced (in order of first ref).
        for key, _n in sorted(order.items(), key=lambda kv: kv[1]):
            seen_keys.add(key)
            body = self._render_def_body(renderer, defs.get(key), key)
            items.append(f'<li id="fn-{key}">{body}</li>')
        # Unreferenced (orphan definitions).
        for key, node in defs.items():
            if key in seen_keys:
                continue
            body = self._render_def_body(renderer, node, key)
            items.append(f'<li id="fn-{key}">{body}</li>')

        if not items:
            return html

        section = (
            '<section class="footnotes"><ol>'
            + "".join(items)
            + "</ol></section>"
        )
        sep = "\n" if html and not html.endswith("\n") else ""
        return html + sep + section

    def _render_def_body(self, renderer, node, key):
        if node is None:
            return f"<p><em>missing footnote: {key}</em></p>"
        inner = renderer.render_children(node)
        return f"<p>{inner}</p>"
