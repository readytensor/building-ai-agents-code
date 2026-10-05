"""md2html — a small Markdown-to-HTML CLI tool.

Public API:

    >>> from md2html import render
    >>> render("# hello")
    '<h1>hello</h1>'

For more control, build the pipeline by hand:

    >>> from md2html import Document
    >>> doc = Document.from_markdown("# hello")
    >>> doc.to_html()
    '<h1>hello</h1>'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .extensions import Extension, default_extensions, resolve_extensions
from .lexer import Lexer
from .parser import Parser
from .renderer import HtmlRenderer

__version__ = "0.1.0"

__all__ = ["Document", "render", "__version__"]


@dataclass
class Document:
    """A parsed Markdown document, ready to be rendered."""

    source: str
    extensions: list[Extension] = field(default_factory=list)

    @classmethod
    def from_markdown(
        cls, source: str, extensions: Iterable[Extension] | None = None
    ) -> "Document":
        exts = list(extensions) if extensions is not None else default_extensions()
        return cls(source=source, extensions=exts)

    def to_html(self) -> str:
        tokens = Lexer(self.source, extensions=self.extensions).tokenize()
        ast = Parser(tokens, extensions=self.extensions).parse()
        return HtmlRenderer(extensions=self.extensions).render(ast)


def render(source: str, extensions: Iterable[Extension] | str | None = None) -> str:
    """Render a Markdown string to HTML.

    `extensions` may be:
      - None: load the default extensions (all of them).
      - A list of Extension instances: use exactly those.
      - A string like "tables,footnotes": resolve names from the registry.
    """
    if isinstance(extensions, str):
        exts = resolve_extensions(extensions)
    elif extensions is None:
        exts = default_extensions()
    else:
        exts = list(extensions)
    return Document.from_markdown(source, extensions=exts).to_html()
