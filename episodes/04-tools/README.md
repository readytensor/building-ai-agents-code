# Tools

**Concept:** how the agent actually does things: a small set of general primitives the
model can compose.

**Additions on top of the loop:** `list_files`, `read`, `write`, `edit`, `grep` for
files and `web_search`, `fetch_url` for the web (alongside `bash`); a tiny `@tool` /
schema helper to remove JSON-schema boilerplate. `list_files` is a cross-platform
listing tool so the agent doesn't have to grope with shell `find`/`ls`/`dir`.
`fetch_url` can save a large file into the sandbox (`save_to`) instead of returning
it, so a big download never fills the context.

**The task:** GitHub emoji shortcodes (`:rocket:` becomes 🚀) as a new md2html
extension, using GitHub's whole list, which only the web has. The fixture pair
`initial/tests/fixtures/emoji.md` / `emoji.html` shows the expected output and fails
until the feature exists.

**Code:**
- `tools.py`: the agent's action space: the tools + the `@tool` decorator. From this
  episode on, new tools land here.
- `agent.py`: the loop, now importing the tools from `tools.py` (the loop itself is
  unchanged except for dispatching by tool name)
- `initial/`: `md2html` with its README and the emoji fixture pair
- `sandbox/`: gitignored, recreated on every run

**Run** (from the repo root):

```bash
python run.py --cwd episodes/04-tools
```

Then try the new feature on a real document:

```bash
python render.py --cwd episodes/04-tools examples/emoji.md --open
```

**Full context:**
- `../../README.md`: companion code repo overview
