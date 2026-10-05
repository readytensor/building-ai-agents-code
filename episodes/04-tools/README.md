# Episode 2: Tools

**Concept:** how the agent actually does things: a small set of general primitives the model can compose.

**This episode's additions on top of Ep 1:** `list_files`, `read`, `write`, `edit`, `grep` tools (alongside `bash`); a tiny `@tool` / schema helper to remove JSON-schema boilerplate. `list_files` is a cross-platform listing tool so the agent doesn't have to grope with shell `find`/`ls`/`dir`.

**Code:**
- `tools.py`: the agent's action space: the five tools + the `@tool` decorator. From Ep 2 on, new tools land here.
- `agent.py`: Ep 1's loop, now importing the tools from `tools.py` (the loop itself is unchanged except for dispatching by tool name)
- `initial/`: `md2html` with one planted bug (an escaped-backtick rendering failure)
- `sandbox/`: gitignored, recreated on every run

**Run:**

```bash
python agent.py
```

**Full context:**
- `../../README.md`: companion code repo overview
