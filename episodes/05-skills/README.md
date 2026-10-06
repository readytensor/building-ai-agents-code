# Skills

**Concept:** knowledge the agent loads when it needs it, instead of carrying it in the
system prompt on every call. A skill holds what the model cannot know on its own:
the team's rules, how its project is extended, how it finishes work.

**Additions on top of Tools:** `skills.py` and a `.skills/` library. A skill is a
folder with a `SKILL.md`: a name, a one-line description of when it applies, and a
body of instructions. Every skill's name and description go into the system prompt
at startup, so the model knows what it can load; `load_skill(name)` returns the body
as a tool result, and only then is it in context. An unused skill costs one line.
The loop itself is unchanged from Tools.

The library holds three skills, each for a reason the model could not supply itself:
`house-typography` (the team's style guide: rules that differ from common defaults
and are written nowhere else), `md2html-extensions` (how the project is extended:
hooks, registry, tests), and `verification` (how the team checks finished work).

**The task:** typographic punctuation (curly quotes, en and em dashes, the ellipsis)
as a new md2html extension, "following the team's style guide". The guide is the
skill. There is no fixture pair for this task: the rules are the knowledge the run
had or did not have. `held_out/` holds the grader's tests of each rule.

**The comparison:** the same task runs twice, once with `.skills/` empty and once with
the library, nothing else changed. Without the guide the model applies the defaults
it knows, which the held-out dash rules fail.

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
- `initial/`: `md2html` with its README, as Tools left it
- `sandbox/`: gitignored, recreated on every run

**Run** (from the repo root):

```bash
python run.py --cwd episodes/05-skills --capture --grade --keep-sandbox
```

`--keep-sandbox` copies the sandbox the run left into its run folder, so the two
runs of the comparison can both be rendered afterwards.

Then try the new feature on a real document:

```bash
python render.py --cwd episodes/05-skills examples/typography.md --open
```

**Full context:**
- `../../README.md`: companion code repo overview
