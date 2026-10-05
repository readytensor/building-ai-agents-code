# Episode 1: The Loop

**Concept:** the minimal agent: a `while` loop calling a single tool until the model stops requesting tool calls.

**This episode's additions:** the loop itself + one `bash` tool + naive stop condition.

**Code:**
- `agent.py`: the agent (~60–80 lines)
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

**Full context:**
- `../../README.md`: companion code repo overview
