"""Small string helpers used across the pipeline."""

from __future__ import annotations

import re

_ESCAPE_MAP = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
}

_ESCAPE_RE = re.compile("|".join(re.escape(c) for c in _ESCAPE_MAP))


def html_escape(text: str, quote: bool = True) -> str:
    """Escape HTML special characters in `text`.

    If `quote` is False, leaves `"` and `'` alone (used inside <pre><code>
    blocks where quotes carry no syntactic meaning in the surrounding HTML).
    """

    def _sub(m: re.Match[str]) -> str:
        ch = m.group(0)
        if not quote and ch in ('"', "'"):
            return ch
        return _ESCAPE_MAP[ch]

    return _ESCAPE_RE.sub(_sub, text)


_SLUG_STRIP_RE = re.compile(r"[^\w\s-]")
_SLUG_HYPHEN_RE = re.compile(r"[-\s]+")


def slugify(text: str) -> str:
    """Generate a URL-safe slug from a heading-ish string."""
    text = text.strip().lower()
    text = _SLUG_STRIP_RE.sub("", text)
    text = _SLUG_HYPHEN_RE.sub("-", text)
    return text.strip("-")


def normalize_whitespace(text: str) -> str:
    """Collapse internal whitespace runs to a single space; strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def expand_tabs(text: str, tabsize: int = 4) -> str:
    """Replace tabs with `tabsize` spaces (we operate in space-indent land)."""
    return text.expandtabs(tabsize)
