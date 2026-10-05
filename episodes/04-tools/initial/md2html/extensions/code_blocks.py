"""Fenced code-block extension.

The core lexer already captures the info string after the opening fence
(see `_consume_fenced_code` in `lexer.py`). All this extension does is
override the renderer for `code_block` nodes so the `<code>` tag carries
a `class="language-xxx"` attribute when a language was given.

No syntax highlighting — just the class attribute, exactly as spec'd.
"""

from __future__ import annotations

import re

from ..utils import html_escape

# A pragmatic allowlist for language slugs we'll accept verbatim. Anything
# else gets squashed through this regex before being inserted into HTML.
_LANG_SAFE_RE = re.compile(r"[^A-Za-z0-9_+\-.]")


def _normalize_lang(info: str) -> str:
    """Normalise the info string to a `language-xxx` slug fragment.

    The info string is the text after the opening fence (e.g. "python" in
    ```` ```python ````). We take only the first whitespace-separated token,
    lowercase it, and strip anything that isn't a friendly identifier char.
    """
    if not info:
        return ""
    token = info.strip().split()[0]
    token = _LANG_SAFE_RE.sub("", token).lower()
    return token


class CodeBlocksExtension:
    name = "code_blocks"

    # No lexer or parser hook needed — the core already captures `info`.
    # We override only the renderer for code_block nodes.

    def render(self, renderer, node):
        if node.kind != "code_block":
            return None
        lang = _normalize_lang(node.attrs.get("lang", ""))
        body = html_escape(node.value, quote=False)
        if lang:
            return f'<pre><code class="language-{lang}">{body}</code></pre>'
        return f"<pre><code>{body}</code></pre>"
