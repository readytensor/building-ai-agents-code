"""Extension registry.

An extension is a small object that may hook into any of the three pipeline
stages. Each hook is optional; the pipeline checks for the attribute before
invoking it.

Hook shape (all optional):

    name: str                                      # registry key
    tokenize_block(lexer) -> bool                  # lexer-stage hook
    parse_inline(parser, text, out) -> bool        # parser inline hook
    render(renderer, node) -> str                  # renderer dispatch hook
    post_render(renderer, html, doc) -> str        # final-pass hook

Tokenizer and inline hooks return True if they consumed input. Renderer hooks
return None to defer to the next handler (or the renderer's built-in).

We keep this deliberately untyped/duck-typed — extensions are simple objects;
no metaclass machinery.
"""

from __future__ import annotations

from typing import Protocol

from .code_blocks import CodeBlocksExtension
from .footnotes import FootnotesExtension
from .tables import TablesExtension


class Extension(Protocol):
    """Structural type for an extension. All hook methods are optional."""

    name: str


# Stable order: tables first (block-level token competes with paragraphs),
# code_blocks second (augments fenced code), footnotes last (post-render
# section append).
_REGISTRY: dict[str, type] = {
    "tables": TablesExtension,
    "code_blocks": CodeBlocksExtension,
    "footnotes": FootnotesExtension,
}


def available_extensions() -> list[str]:
    return list(_REGISTRY.keys())


def default_extensions() -> list[Extension]:
    """Return one fresh instance of every shipped extension."""
    return [cls() for cls in _REGISTRY.values()]


def resolve_extensions(spec: str) -> list[Extension]:
    """Resolve a comma-separated list of extension names.

    Empty string or "none" disables all extensions.
    Unknown names raise ValueError with a list of valid ones.
    """
    spec = spec.strip()
    if not spec or spec.lower() == "none":
        return []
    names = [n.strip() for n in spec.split(",") if n.strip()]
    out: list[Extension] = []
    for name in names:
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown extension {name!r}. Available: {', '.join(_REGISTRY)}"
            )
        out.append(_REGISTRY[name]())
    return out


__all__ = [
    "Extension",
    "available_extensions",
    "default_extensions",
    "resolve_extensions",
]
