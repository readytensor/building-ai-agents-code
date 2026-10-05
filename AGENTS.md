# AGENTS.md

Guidance for AI coding agents (and curious humans) working in this repository.

This is the companion code for the **Building AI Agents** video series. It builds one coding agent from scratch, one capability per episode: a plain `while` loop with a single tool first, then tools, skills, subagents and verification, using a small Markdown-to-HTML CLI called `md2html` as the example project to work on. No frameworks: just Python and a model API.

For the full series narrative, the episode table, and setup/quickstart, read [`README.md`](./README.md). This file covers how to *work* in the repo, not what the series teaches.

## Finding your way around

```
building-ai-agents-code/
├── episodes/
│   ├── 03-loop/
│   │   ├── agent.py           # the episode's agent: start here
│   │   ├── system_prompt.md   # the agent's system prompt (shared core across episodes)
│   │   ├── initial/           # pristine starting copy of the md2html project
│   │   └── sandbox/           # where the agent works (recreated every run)
│   ├── 04-tools/  05-skills/  06-subagents/   (03-compaction/ and 04-working-memory/: earlier plan, to be removed)
├── examples/             # the FINISHED md2html (every feature) + a sample doc that uses them all
├── eval/                 # evaluation harness (SWE-bench Verified + the episode tasks)
├── run.py                # optional harness to record a run
├── capture.py            # terminal recorder used by run.py --capture
└── requirements.txt
```

Each episode is **self-contained**: `cd` into it and run `python agent.py`. Episodes build on each other conceptually, but each one is a complete, standalone program: there is no shared library and no branch switching. The interesting comparison is the diff:

```bash
diff episodes/03-loop/agent.py episodes/04-tools/agent.py   # what one idea added
```

`agent.py` is always the entry point. The early episodes are a single file; as the agent grows, later episodes split the supporting pieces into a few small modules next to it (`tools.py`, `compaction.py`, `planning.py`, `skills.py`), so `agent.py` stays focused on the loop.

## Ground rules when working here

- **Never modify `initial/`.** It is the pristine template the agent starts from. Every `agent.py` begins by wiping `sandbox/` and copying `initial/` into it, so each run starts from an identical clean state. If you change `initial/`, you change the experiment.
- **`sandbox/` is ephemeral.** It is recreated on every run and is gitignored. Don't keep anything there you care about, and don't be surprised when it resets.
- **Stay inside one episode.** A change for Episode 3 belongs in `episodes/03-compaction/`. Don't edit one episode's files to fix another, and don't let one episode's `agent.py` reach into another's directory.
- **The diff between episodes is the lesson.** When you add or change something, keep the *delta* from the previous episode small and legible: that delta is the teaching point, not just the end state.
- **`bash` is not sandboxed.** The file tools are contained to `sandbox/`, but `bash` only *starts* there; it runs with the user's full permissions. See "A note on safety" in [`README.md`](./README.md) before pointing the agent at anything beyond the scripted episode tasks.
- **The system prompt is one shared artifact.** Every episode's `system_prompt.md` carries the same core text; later episodes add only the section for the mechanism they introduce, and `eval/system_prompt.md` matches Episode 5 exactly. Never edit one copy alone: change all of them together (a drift test in `eval/tests/` fails otherwise).

## Verify your work

`md2html` ships with a pytest suite. Use it, because that's the whole point of working against a tested codebase.

```bash
cd episodes/04-tools/sandbox && python -m pytest -q
```

**"Done" means the tests pass, not that the agent (or you) said so.** Confirm with command output before claiming a change works. Several episodes are built around exactly this discipline: verifying with tests rather than trusting a self-assessment.

The eval harness has its own offline suite (fake agents, no API calls, runs in seconds):

```bash
python -m pytest eval/tests -q
```

## Code conventions

This is a **teaching repository**. The code is the artifact students learn from, so optimize for being read, not for being clever.

- **Write code that's easy to follow.** Clear names, obvious control flow, short functions. Not unnecessarily verbose (but not production-dense code packed for performance or generality either). If a student has to pause to decode a line, rewrite it.
- **One idea per episode.** Each episode adds exactly one concept on top of the last. Keep changes minimal and focused; resist adding scaffolding or features the episode doesn't need.
- **No frameworks.** The series builds agents from primitives on purpose. Reach for the standard library and the model API, not an agent framework.
- **Provider-portable LLM calls.** The code uses the `openai` Python package against the **Chat Completions API** so it runs against any OpenAI-compatible endpoint (set `LLM_BASE_URL` / `LLM_AGENT_MODEL` in `.env`). Avoid provider-specific features: keep it portable.

## Out of scope

This series is about the architectural core of how agents work: the loop, tools, context management, planning, skills, and multi-agent topology. Production ops, durable execution, full guardrails, framework reviews, and model training/RL are each their own topic and deliberately left out. Don't pull them in.
