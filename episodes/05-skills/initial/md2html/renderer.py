"""AST -> HTML via visitor methods.

The renderer dispatches on `node.kind` to a `visit_<kind>` method. Extensions
get two hooks:

- `render(renderer, node) -> str | None` — called *before* the built-in
  dispatch. Returning a string short-circuits; returning None falls through.
- `post_render(renderer, html, doc_root) -> str` — called once at the end
  to massage the final document (footnotes uses this to append its
  collected definitions).
"""

from __future__ import annotations

from .parser import Node
from .utils import html_escape


class HtmlRenderer:
    """Walk the AST and emit HTML."""

    def __init__(self, extensions: list | None = None) -> None:
        self.extensions = extensions or []
        # Renderer-wide state extensions can scribble on (e.g. footnotes
        # uses self.state["footnotes"] to collect refs across the doc).
        self.state: dict = {}
        for ext in self.extensions:
            hook = getattr(ext, "init_state", None)
            if hook is not None:
                hook(self)

    # -- entry point --------------------------------------------------------

    def render(self, root: Node) -> str:
        # Top-level: dispatch through render_node so visit_document (which
        # newline-joins block children) handles the layout.
        body = self.render_node(root)
        for ext in self.extensions:
            hook = getattr(ext, "post_render", None)
            if hook is not None:
                body = hook(self, body, root)
        return body

    # -- dispatch -----------------------------------------------------------

    def render_node(self, node: Node) -> str:
        # Extension hook first.
        for ext in self.extensions:
            hook = getattr(ext, "render", None)
            if hook is None:
                continue
            out = hook(self, node)
            if out is not None:
                return out

        method = getattr(self, f"visit_{node.kind}", None)
        if method is None:
            # Unknown node — render children inline as a best-effort fallback.
            return self.render_children(node)
        return method(node)

    def render_children(self, node: Node) -> str:
        return "".join(self.render_node(c) for c in node.children)

    # -- block visitors -----------------------------------------------------

    def visit_document(self, node: Node) -> str:
        # Block-level children get newline-separated; nothing wraps the doc.
        parts: list[str] = []
        for child in node.children:
            parts.append(self.render_node(child))
        return "\n".join(p for p in parts if p)

    def visit_heading(self, node: Node) -> str:
        level = node.attrs.get("level", 1)
        return f"<h{level}>{self.render_children(node)}</h{level}>"

    def visit_paragraph(self, node: Node) -> str:
        return f"<p>{self.render_children(node)}</p>"

    def visit_hr(self, node: Node) -> str:
        return "<hr/>"

    def visit_code_block(self, node: Node) -> str:
        # Core renderer: no language class. The code_blocks extension
        # overrides this via its `render` hook when enabled.
        body = html_escape(node.value, quote=False)
        return f"<pre><code>{body}</code></pre>"

    def visit_blockquote(self, node: Node) -> str:
        inner = "\n".join(self.render_node(c) for c in node.children if c.kind != "document")
        if not inner:
            inner = self.render_children(node)
        return f"<blockquote>{inner}</blockquote>"

    def visit_list(self, node: Node) -> str:
        tag = "ol" if node.attrs.get("ordered") else "ul"
        attrs = ""
        start = node.attrs.get("start")
        if tag == "ol" and start not in (None, 1):
            attrs = f' start="{start}"'
        items = "".join(self.render_node(c) for c in node.children)
        return f"<{tag}{attrs}>{items}</{tag}>"

    def visit_list_item(self, node: Node) -> str:
        # If the item contains a nested list as one of its children, render
        # tightly: <li>text<ul>...</ul></li>.
        return f"<li>{self.render_children(node)}</li>"

    # -- inline visitors ----------------------------------------------------

    def visit_text(self, node: Node) -> str:
        return html_escape(node.value)

    def visit_emph(self, node: Node) -> str:
        return f"<em>{self.render_children(node)}</em>"

    def visit_strong(self, node: Node) -> str:
        return f"<strong>{self.render_children(node)}</strong>"

    def visit_emph_strong(self, node: Node) -> str:
        return f"<strong><em>{self.render_children(node)}</em></strong>"

    def visit_code(self, node: Node) -> str:
        return f"<code>{html_escape(node.value, quote=False)}</code>"

    def visit_link(self, node: Node) -> str:
        href = html_escape(node.attrs.get("href", ""))
        title = node.attrs.get("title", "")
        title_attr = f' title="{html_escape(title)}"' if title else ""
        return f'<a href="{href}"{title_attr}>{self.render_children(node)}</a>'

    def visit_image(self, node: Node) -> str:
        src = html_escape(node.attrs.get("src", ""))
        alt = html_escape(node.value)
        title = node.attrs.get("title", "")
        title_attr = f' title="{html_escape(title)}"' if title else ""
        return f'<img src="{src}" alt="{alt}"{title_attr}/>'

    def visit_linebreak(self, node: Node) -> str:
        return "<br/>"
