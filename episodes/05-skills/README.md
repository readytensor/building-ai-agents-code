# Skills

**Concept:** instructions the agent loads when it needs them, instead of carrying
every procedure in its system prompt on every call.

**Additions on top of Tools:** `skills.py` and a `.skills/` library. A skill is a
folder with a `SKILL.md`: a name, a one-line description of when it applies, and a
body of instructions. Every skill's name and description go into the system prompt
at startup, so the model knows what it can load; `load_skill(name)` returns the body
as a tool result, and only then is it in context. An unused skill costs one line.
The loop itself is unchanged from Tools.

The library holds three general skills, written for any codebase (the same agent
later runs on SWE-bench): `implementing-a-feature`, `verification`, `fixing-a-bug`.
On this episode's task the first two load, at the start and before the finish; the
third stays unloaded.

**The task:** GitHub-flavored alerts (`> [!NOTE]` and the other four types) as a new
md2html extension. The fixture pair `initial/tests/fixtures/github_alerts.md` /
`github_alerts.html` shows the expected output and fails until the feature exists.

**Code:**
- `skills.py` (**this episode's addition**): `skills_index`, `load_skill`, and the
  `SKILL.md` parser
- `.skills/`: the skill library, next to the agent's code (it is agent
  infrastructure, not part of the project the agent works on, so it is outside
  `initial/` and the sandbox; `load_skill` is the only way to it)
- `agent.py`: the loop, unchanged except that the system prompt ends with the skills
  index and `load_skill` is one more tool
- `system_prompt.md`: the shared core plus a Skills section that names the moments
  to load a skill
- `tools.py`: carried forward from Tools unchanged
- `grade.py` and `held_out/`: tests the agent never sees, run against the sandbox
  after a run (`python grade.py` from this folder)
- `initial/`: `md2html` with its README and the alerts fixture pair
- `sandbox/`: gitignored, recreated on every run

**Run** (from the repo root):

```bash
python run.py --cwd episodes/05-skills
```

Then try the new feature on a real document:

```bash
python render.py --cwd episodes/05-skills examples/alerts.md --open
```

**Full context:**
- `../../README.md`: companion code repo overview
