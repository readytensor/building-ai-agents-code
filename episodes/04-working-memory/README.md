# Episode 4: Working Memory

**Concept:** give the agent durable, self-maintained state that survives compaction.
On a long task the agent needs to track what's done and what's next, but anything it
records in the transcript gets summarized away by Ep 3's compaction. The fix is a slot
the agent maintains in *state* and the loop re-injects into the system prompt every
turn, so it stays in front of the model and compaction can't touch it. A plan is the
first thing worth keeping there.

**This episode's additions on top of Ep 3:**
- `write_plan(steps)`: a Claude Code TodoWrite-style structured plan that lives in
  agent state (the `CURRENT_PLAN` global), not the transcript. The agent writes and
  updates it as work progresses.
- The **dynamic system prompt**: each iteration the loop rebuilds the system message
  as `[stable base + current plan]`, so the plan and its running status are always
  visible and survive compaction untouched. This same mechanism carries loaded-skill
  bodies in Ep 5.

`write_plan` is the worked instance of the durable slot; the same mechanism would just
as well hold key analysis results, hard constraints, or a running summary.

**Code** (structured like Ep 3: the loop, the tools, and each mechanism in its own
file):
- `agent.py`: the agent loop; differs from Ep 3 only where the new mechanism plugs in
  (the planning import, the extended tool registry, and rebuilding the system prompt
  with the current plan each iteration)
- `planning.py` (**this episode's addition**): `write_plan`, the `CURRENT_PLAN` state,
  and the dynamic system-prompt mechanism (kept in one file the way Ep 3 kept
  compaction in `compaction.py`)
- `tools.py`, `compaction.py`: carried forward from Ep 3 unchanged
- `initial/`: `md2html` (the toy codebase) with a test fixture for reference-style
  markdown links; the fixture fails until the feature is implemented
- `sandbox/`: gitignored, recreated on every run

**Run:**

```bash
python agent.py
```

The agent's task: implement reference-style markdown link support so all tests pass.
See `initial/tests/fixtures/reference_style_links.md` and `.html` for the spec by
example.

**Full context:**
- `../../README.md`: companion code repo overview
