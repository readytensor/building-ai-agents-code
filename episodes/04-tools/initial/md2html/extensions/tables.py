"""GitHub-flavored pipe-table extension.

Hooks:

- `tokenize_block(lexer)` — when the current line looks like a table header
  followed by a separator row, consume the whole table into a single
  TK_TABLE token.
- `parse_block(parser, tok)` — turn TK_TABLE tokens into TableNode AST.
- `render(renderer, node)` — emit <table>…</table>.

Table syntax:

    | col1 | col2 |
    |------|------|
    | a    | b    |

Alignment via colons in the separator row:
`|:---|---:|:---:|` → left, right, center.
"""

from __future__ import annotations

import re

from ..lexer import Token
from ..parser import Node

TK_TABLE = "table"

# A "row" line looks like `|...|...|` or `a | b | c` with at least one pipe.
_RE_ROW = re.compile(r"^[ \t]*\|?(.+?)\|?[ \t]*$")
# The separator row: each cell is dashes, optionally bracketed by colons,
# separated by pipes.
_RE_SEP_CELL = re.compile(r"^\s*(:?-{2,}:?)\s*$")


def _split_row(line: str) -> list[str]:
    """Split a table row into cells. Pipes preceded by a backslash are
    treated as literal `|` inside a cell.
    """
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            buf.append("|")
            i += 2
            continue
        if c == "|":
            cells.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    cells.append("".join(buf).strip())
    return cells


def _parse_alignment(sep_line: str) -> list[str] | None:
    """Parse the separator row. Returns a list of alignments (one per
    column) or None if `sep_line` isn't a valid separator.
    """
    cells = _split_row(sep_line)
    if not cells:
        return None
    aligns: list[str] = []
    for cell in cells:
        if not _RE_SEP_CELL.match(cell):
            return None
        left = cell.startswith(":")
        right = cell.endswith(":")
        if left and right:
            aligns.append("center")
        elif right:
            aligns.append("right")
        elif left:
            aligns.append("left")
        else:
            aligns.append("")
    return aligns


def _looks_like_table_header(line: str) -> bool:
    """A row line containing at least one pipe — required for the header."""
    return "|" in line and bool(_RE_ROW.match(line))


class TablesExtension:
    name = "tables"

    # ------------------------------------------------------------------
    # Lexer hook
    # ------------------------------------------------------------------

    def tokenize_block(self, lexer) -> bool:
        line = lexer.line()
        if not _looks_like_table_header(line):
            return False
        nxt = lexer.peek()
        if not nxt:
            return False
        aligns = _parse_alignment(nxt)
        if aligns is None:
            return False

        header_cells = _split_row(line)
        if len(header_cells) != len(aligns):
            return False

        lexer.advance()  # consume header
        lexer.advance()  # consume separator

        rows: list[list[str]] = []
        while not lexer.eof():
            row_line = lexer.line()
            if not row_line.strip() or "|" not in row_line:
                break
            cells = _split_row(row_line)
            # Pad / truncate to column count.
            if len(cells) < len(aligns):
                cells = cells + [""] * (len(aligns) - len(cells))
            elif len(cells) > len(aligns):
                cells = cells[: len(aligns)]
            rows.append(cells)
            lexer.advance()

        lexer.emit(
            Token(
                TK_TABLE,
                "",
                {"header": header_cells, "align": aligns, "rows": rows},
            )
        )
        return True

    def breaks_paragraph(self, line: str) -> bool:
        # If we're in a paragraph and we see a possible header-row-like
        # line, don't break — we can't detect a table from one line alone.
        # Returning False here is correct; the lexer's table detection
        # operates only at block boundaries (after blank lines).
        return False

    # ------------------------------------------------------------------
    # Parser hook
    # ------------------------------------------------------------------

    def parse_block(self, parser, tok):
        if tok.kind != TK_TABLE:
            return None
        parser.advance()
        header = tok.attrs["header"]
        rows = tok.attrs["rows"]
        align = tok.attrs["align"]

        header_nodes = [
            Node("table_cell", children=parser._parse_inline(c), attrs={"header": True, "align": a})
            for c, a in zip(header, align)
        ]
        row_nodes = []
        for row in rows:
            cells = [
                Node(
                    "table_cell",
                    children=parser._parse_inline(c),
                    attrs={"header": False, "align": a},
                )
                for c, a in zip(row, align)
            ]
            row_nodes.append(Node("table_row", children=cells))

        return Node(
            "table",
            children=[Node("table_header", children=header_nodes)] + row_nodes,
            attrs={"align": align},
        )

    # ------------------------------------------------------------------
    # Renderer hook
    # ------------------------------------------------------------------

    def render(self, renderer, node):
        if node.kind == "table":
            head = node.children[0]
            body_rows = node.children[1:]
            head_html = self._render_header(renderer, head)
            body_html = "".join(self._render_row(renderer, r) for r in body_rows)
            body_block = f"<tbody>{body_html}</tbody>" if body_html else ""
            return f"<table><thead>{head_html}</thead>{body_block}</table>"

        if node.kind == "table_cell":
            tag = "th" if node.attrs.get("header") else "td"
            align = node.attrs.get("align", "")
            attr = f' style="text-align:{align}"' if align else ""
            content = renderer.render_children(node)
            return f"<{tag}{attr}>{content}</{tag}>"

        if node.kind == "table_row":
            return "<tr>" + "".join(renderer.render_node(c) for c in node.children) + "</tr>"

        if node.kind == "table_header":
            return "<tr>" + "".join(renderer.render_node(c) for c in node.children) + "</tr>"

        return None

    def _render_header(self, renderer, head_node):
        return renderer.render_node(head_node)

    def _render_row(self, renderer, row_node):
        return renderer.render_node(row_node)
