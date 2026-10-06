# The Loop

**Concept:** the minimal agent: a `while` loop calling a single tool until the model stops requesting tool calls.

**This episode's additions:** the loop itself + one `bash` tool + naive stop condition.

**Code:**
- `agent.py`: the agent: the `bash` tool, the loop in `run_agent()`, and `main()`
- `initial/`: pristine `md2html` starting state (committed)
- `sandbox/`: agent's working dir (gitignored, recreated on every run)

**Run:**

```bash
python agent.py
```

After a run, inspect what the agent did:

```bash
diff -r initial sandbox
```

Then try the new feature on a real document (from the repo root):

```bash
python render.py --cwd episodes/03-loop examples/toc.md --open
```

**Full context:**
- `../../README.md`: companion code repo overview
