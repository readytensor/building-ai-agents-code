# Episode 6: Subagents

**Concept:** one agent becomes many. An *orchestrator* breaks a task into independent subtasks and `delegate`s each to a fresh worker agent; independent workers run in parallel, and a dedicated *verifier* owns completion. It's Ep 5's loop, made reentrant, plus one new tool and one new config primitive.

**This episode's additions on top of Ep 5:**
- `delegate(task, agent_type)`: spawns a fresh worker agent configured by `.agents/<agent_type>.md` (its toolset + any preloaded skills) and returns the worker's result.
- A `.agents/<name>.md` config primitive (YAML frontmatter + markdown body), parsed by the same helper as Ep 5's `SKILL.md`.
- `run_agent(task, agent_type)`: Ep 5's loop, extracted into a reentrant function. The orchestrator and every worker run through the *same* function, parameterized by an `AgentConfig`.
- Parallel dispatch (a `ThreadPoolExecutor`): when one assistant turn emits multiple `delegate` calls, they fan out concurrently and return together.

Roles are enforced by **toolset, not exhortation**: the orchestrator gets no codebase-mutation tools (no `read`/`write`/`edit`/`bash`/`grep`; all work goes through workers), and the `verifier` has no `write`/`edit`. Completion is **verifier-owned**: the orchestrator reaches its natural stop and returns once the verifier confirms a clean test pass (no "done" tool).

**Code** (structured like Ep 5: the loop, the tools, and each mechanism in its own file):
- `agent.py` (**this episode's addition**): the orchestrator + worker runtime — the `delegate` tool, the `.agents/<name>.md` loader, the reentrant `run_agent`, and the parallel dispatcher
- `skills.py`, `planning.py`: carried forward from Ep 5, but adapted so their state is **per-call** (each worker owns its own plan + loaded skills via closures) instead of module-global — so concurrent workers never share state
- `tools.py`, `compaction.py`: carried forward from Ep 5
- `.agents/`: the worker configs, at the episode root alongside the agent's code (agent infrastructure, not part of the toy codebase, so not inside `initial/` or the sandbox):
  - `implementer.md`: full toolset, `verification` skill preloaded
  - `verifier.md`: read/test tools only (no `write`/`edit`)
- `.skills/`: `research` and `verification`, carried forward from Ep 5; at the episode root alongside the agent's code (agent infrastructure, not part of the toy codebase, so not inside `initial/`)
- `initial/`: `md2html` with three independent test fixtures (one per feature below); each fails until its feature is implemented
- `sandbox/`: gitignored, recreated on every run
- `messages.jsonl`: gitignored run artifact, written as the run goes. Each agent keeps its own append-only history and compacts non-destructively (Ep 3's mechanism, per worker), but they all append to this one file with their label attached, so the whole tree's transcript lands in one place. Filter by `agent` to replay a single worker

**Run:**

```bash
python agent.py
```

The agent's task: add three independent GitHub-flavored-markdown features to `md2html` at once (**strikethrough**, **task lists**, and **autolinks**). Because the three don't depend on each other, the orchestrator's value shows up directly in the trajectory (concurrent `delegate` calls in a single turn). See the spec-by-example fixtures: `initial/tests/fixtures/{strikethrough,task_lists,autolinks}.md` and their `.html` pairs.

**Full context:**
- `../../README.md`: companion code repo overview
