# Building AI Agents

Build a working coding agent from scratch: from a plain `while` loop with a single tool to skills, subagents, and checks the work must pass before the agent may stop, and then test it on a public benchmark. No frameworks: just Python and a model API.

This is the companion code for the **Building AI Agents** video series. Each episode adds one idea, in code, on top of the last, and the diff between one episode and the next is the lesson.

## Who this is for

Engineers comfortable with Python and calling an LLM API. No prior agent-building experience required. By the end you'll be able to build a basic agent, extend it deliberately, debug it when it breaks, and critically evaluate any agent system you encounter.

The worked example is a **coding agent**, the cleanest domain to learn in: a tight feedback loop and a small tool surface. The same ideas carry over to research agents, browser automation, and data pipelines.

## The series

The videos are numbered straight through, in three parts.

| #   | Part               | Episode         | The question                                  | What the agent gains                                                                  |
| --- | ------------------ | --------------- | --------------------------------------------- | ------------------------------------------------------------------------------------- |
| 01  | Getting started    | Series Overview | What will we build?                           | Nothing yet: the loop, the project, and the plan                                      |
| 02  | Getting started    | Project Setup   | How do I run it?                              | Nothing yet: the repository, and how to run the code                                  |
| 03  | Building the agent | The Loop        | What is an agent?                             | A `while` loop and one `bash` tool                                                    |
| 04  | Building the agent | Tools           | How does it actually do things?               | General tools (`read`, `write`, `edit`, `grep`, `list_files`) and a small `@tool` helper |
| 05  | Building the agent | Skills          | How does it reach beyond its fixed toolkit?   | Capabilities loaded only when needed (`list_skills`, `load_skill`, `SKILL.md`)       |
| 06  | Building the agent | Subagents       | When is one agent not enough?                 | `delegate`, worker configs, and parallel workers, each with its own context           |
| 07  | Building the agent | Verification    | How do we know the work is finished and good? | A completion gate at the stop, and grading outside the loop                           |
| 08+ | The benchmark      | Setup, then results | Does it hold up on real code?             | The same agent on SWE-bench Verified (`eval/`)                                        |

Each episode in "Building the agent" follows the same rhythm: one question, one limitation, one addition in code, one before/after.

> [!NOTE]
> The code is being updated to this arc one episode at a time. `03-loop` and `04-tools` already carry their video numbers. `03-compaction` and `04-working-memory` are from an earlier plan and will be removed, and `05-skills` and `06-subagents` still include that plan's compaction and planning code until their episodes are updated.

## How the code is organized

```
building-ai-agents-code/
├── episodes/
│   ├── 03-loop/
│   │   ├── agent.py           # the episode's agent: start here
│   │   ├── system_prompt.md   # the agent's system prompt (prompt text is config, not code)
│   │   ├── initial/           # pristine starting copy of the example project
│   │   └── sandbox/           # where the agent works (recreated every run)
│   ├── 04-tools/              # + tools.py (each later episode adds one file per mechanism)
│   ├── 03-compaction/         # earlier plan, to be removed: + compaction.py, held_out/ + grade.py
│   ├── 04-working-memory/     # earlier plan, to be removed: + planning.py
│   ├── 05-skills/             # + skills.py and a .skills/ library
│   └── 06-subagents/          # + .agents/ worker configs
├── examples/
│   ├── md2html/             # the FINISHED tool (every feature) you build up to
│   └── about-the-series.md  # a sample document that exercises every feature
├── eval/                # evaluation harness: the agent on SWE-bench Verified + the episode tasks
├── run.py               # optional harness to record a run (see below)
├── capture.py           # terminal recorder used by run.py --capture
├── requirements.txt
└── .env.example
```

Each episode is **self-contained**: `cd` into it and run `python agent.py`. No branch switching.

Each `agent.py` is also **importable**: the loop lives in `run_agent(client, model, system, tools, task)`, and importing the module has no side effects. `main()` owns everything that touches the world (the sandbox reset, the client, the telemetry files), so you can reuse the loop in your own code:

```python
from agent import run_agent, make_client, SYSTEM, TOOLS
```

**`system_prompt.md`.** Each episode's system prompt is a markdown file next to `agent.py`, loaded in one line. The prompt shares a common core across every episode; later episodes add only the section for the mechanism they introduce (a plan section in Episode 4, a skills section in Episode 5). Diff two of them to see exactly what an episode taught the agent.

**`eval/`.** A separate harness that runs the finished agent against real problems: SWE-bench Verified instances (with official Docker grading) and the series' own episode tasks. See [`eval/README.md`](./eval/README.md).

**`initial/` → `sandbox/`.** Every `agent.py` begins by wiping `sandbox/` and copying `initial/` into it, so each run starts from an identical clean state. `initial/` is never modified; the agent only works inside `sandbox/`. After a run, see what it changed:

```bash
diff -r initial sandbox
```

**The diff between episodes is the lesson.** Compare two agents to see exactly what each idea added:

```bash
diff episodes/03-loop/agent.py episodes/04-tools/agent.py
```

## Recording a run (optional)

`python agent.py` runs the agent on its own. If you want to capture what happened (to compare runs or inspect the agent's path), use the `run.py` harness instead (from the repo root):

```bash
python run.py --cwd episodes/03-loop            # record the tool-call sequence to logs/<timestamp>/
python run.py --cwd episodes/03-loop --capture  # also save the full terminal output
python run.py --cwd episodes/03-compaction --capture -g   # and grade the run afterwards (see below)
```

Each run gets its own timestamped folder under the episode's `logs/`, so you can run the same task repeatedly and compare how the agent's path and tool-call count vary from run to run. `capture.py` is the underlying terminal recorder and also works standalone on any command (e.g. `python capture.py -- pytest -q`).

## Grading a run

The agent verifies its own work with what it can see: the failing fixture, the tests it writes, the project's suite. Grading is a different act: judging the finished run against **held-out tests the agent never saw**. The `03-compaction` folder introduces the pattern with a `held_out/` folder at the episode root (never copied into the sandbox) and a small `grade.py` that injects those tests after a run and re-runs the suite:

```bash
cd episodes/03-compaction
python agent.py     # the run
python grade.py     # the judgment: GRADE: PASS / FAIL
```

Or in one step with the harness: `python run.py -g`, which also saves the verdict as `grade.log` in the run's folder. A run that passes everything the agent could see but fails the held-out tests fit the examples it was given without implementing the rule behind them. That distinction (the agent verifies, the harness grades) comes back in `05-skills` and `06-subagents`, and at full scale in `eval/`.

## The example project: `md2html`

Every episode points the agent at the same small codebase, `md2html`, a Markdown-to-HTML CLI with real module boundaries (lexer → parser → renderer → extensions → CLI) and a pytest suite. Small enough to follow, structured enough that each episode's task lands on a real seam, and the tests let the agent verify its own work instead of just claiming success.

The **finished** version of that tool, with the features the agent builds across the series (reference links, GitHub alerts, strikethrough, task lists, autolinks, and the rest), lives in [`examples/md2html/`](./examples/md2html/). It's a complete, self-contained project (`pytest` is green). Try it on the sample document, which uses every feature:

```bash
cd examples/md2html
python -m md2html ../about-the-series.md --standalone   # writes examples/about-the-series.html, then open it in a browser
```

With no `-o`, the HTML is written next to the source file (`examples/about-the-series.html`). `--standalone` wraps the output in a full HTML page with a built-in stylesheet; without it, `md2html` emits just the body fragment, the usual contract for a Markdown converter.

## Provider portability

The code uses the `openai` Python package against the **Chat Completions API**, so it runs against any OpenAI-compatible endpoint; just set `LLM_BASE_URL` and `LLM_AGENT_MODEL` in `.env`. Compatible providers include OpenAI, Groq, Together, Mistral, DeepSeek, Ollama, vLLM, and OpenRouter. We avoid provider-specific features so the same code stays portable.

## Scope

This series is about the **architectural core** of how agents work: the loop, tools, skills, subagents, and verifying the agent's work. Deliberately out of scope (each its own topic) are production ops, durable execution, full guardrails, framework reviews, and model training/RL.

## License

MIT. See [LICENSE](./LICENSE). Use the code freely in your own projects and experiments.
