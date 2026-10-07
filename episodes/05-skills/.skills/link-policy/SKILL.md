---
name: link-policy
description: The team's rules for links in rendered HTML, which schemes are allowed and the attributes every link carries. Use when a task adds a way to write links or changes how links are rendered.
---

# Link policy

Every link md2html renders follows these rules, however it was written: an
inline link, a reference link, or an autolink. They are the team's own, so
apply them as written rather than a common default.

## Allowed addresses

- `http://` and `https://` addresses, `mailto:` addresses, relative paths, and
  `#fragment` links are allowed.
- Anything else (`javascript:`, `data:`, `file:`, `ftp:` and the rest) is not a
  link. Render its text as plain, escaped text, with no `<a>` element.

## Attributes

- An external link (an `http://` or `https://` address) carries
  `class="ext"` and `rel="noopener nofollow"`.
- A `mailto:` link, a relative path, or a `#fragment` link carries neither.
- Attributes come in this order: `href`, `title` (only when the link has one),
  `class`, `rel`.

## Examples

| Markdown | HTML |
|---|---|
| `[docs](https://example.com/docs)` | `<a href="https://example.com/docs" class="ext" rel="noopener nofollow">docs</a>` |
| `[intro](#intro)` | `<a href="#intro">intro</a>` |
| `[write to us](mailto:team@example.com)` | `<a href="mailto:team@example.com">write to us</a>` |
| `[click](javascript:alert(1))` | `click` |
