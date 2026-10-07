---
name: markdown-spec
description: md2html follows the GitHub Flavored Markdown spec; where to read it, and how its examples become tests. Use when a task implements or changes syntax the spec defines, such as tables, strikethrough, task lists, autolinks or reference links.
---

# The Markdown spec

md2html follows the GitHub Flavored Markdown spec, version 0.29-gfm, which is
CommonMark plus GitHub's extensions. Work from the spec's text, not from memory:
GitHub's rules differ from other Markdown flavors in their edge cases, and the
spec's examples are how this project defines correct.

## Reading it

- The spec is one page: `https://github.github.com/gfm/`. It is about 500 KB,
  more than a fetch returns in full, so fetch it with `save_to` into
  `.spec/gfm.html` and search that file for the section you need.
- Delete the `.spec/` folder before you finish.

| Feature | Section id on the page |
|---|---|
| Strikethrough | `strikethrough-extension-` |
| Task list items | `task-list-items-extension-` |
| Autolinks (bare URLs and `www.`) | `autolinks-extension-` |
| Link reference definitions | `link-reference-definitions` |
| Reference links | `full-reference-link`, `collapsed-reference-link`, `shortcut-reference-link` |
| Tables | `tables-extension-` |

## Using it

- Read the whole section, including every numbered example. Each example pairs
  Markdown with the HTML the spec expects.
- Turn the examples into tests: a unit test per example, or a fixture pair in
  `tests/fixtures/`. Where md2html's existing output style differs from the
  spec's HTML only in whitespace or attribute order, match md2html's style.
- In your final summary, name the spec version and the sections you followed.
