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
| 04  | Building the agent | Tools           | How does it actually do things?               | General tools for files (`read`, `write`, `edit`, `grep`, `list_files`) and the web (`web_search`, `fetch_url`), and a small `@tool` helper |
| 05  | Building the agent | Skills          | How does it reach beyond its fixed toolkit?   | Capabilities loaded only when needed (`list_skills`, `load_skill`, `SKILL.md`)       |
| 06  | Building the agent | Subagents       | When is one agent not enough?                 | `delegate`, worker configs, and parallel workers, each with its own context           |
| 07  | Building the agent | Verification    | How do we know the work is finished and good? | A completion gate at the stop, and grading outside the loop                           |
| 08+ | The benchmark      | Setup, then results | Does it hold up on real code?             | The same agent on SWE-bench Verified (`eval/`)                                        |

Each episode in "Building the agent" follows the same rhythm: one question, one limitation, one addition in code, one before/after.

> [!NOTE]
> The code is being updated to this arc one episode at a time, so some folders in `episodes/` are still from an earlier version of the series.

## A note on safety

The agent runs real shell commands with your user account's permissions. The file tools (`read`, `write`, `edit`, `grep`, `list_files`) are restricted to the `sandbox/` folder and cannot escape it, but `bash` is not: the sandbox is only its starting directory, and nothing prevents a command from using `cd ..` or an absolute path. The model decides what commands to run, so treat every run as untrusted.

Following along with the episode tasks as written is low risk, and that is how we run it. If you point the agent at your own tasks, your own repositories, or open-ended experiments, run it inside a Docker container or a throwaway VM, not directly on a machine you care about. Real agent products solve this with OS-level sandboxes and containers; that layer is out of scope for this series on purpose.

## Quickstart

```bash
git clone https://github.com/readytensor/building-ai-agents-code
cd building-ai-agents-code
```

Set up a virtual environment and install the dependencies (Python 3.11+):

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then add your provider's API key
```

Using `uv`? `uv venv && uv pip install -r requirements.txt`.

Run an episode:

```bash
cd episodes/03-loop
python agent.py
```

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
│   ├── 05-skills/             # + skills.py and a .skills/ library
│   └── 06-subagents/          # + .agents/ worker configs
├── examples/
│   ├── md2html/             # the FINISHED tool (every feature) you build up to
│   ├── about-the-series.md  # a sample document that exercises every feature
│   └── toc.md, emoji.md     # one sample document per feature the agent builds
├── eval/                # evaluation harness: the agent on SWE-bench Verified + the episode tasks
├── run.py               # optional harness to record a run (see below)
├── render.py            # render a Markdown file with an episode's md2html (see below)
├── capture.py           # terminal recorder used by run.py --capture
├── requirements.txt
└── .env.example
```

Each episode is **self-contained**: `cd` into it and run `python agent.py`. No branch switching.

Each `agent.py` is also **importable**: the loop lives in `run_agent(client, model, system, tools, task)`, and importing the module has no side effects. `main()` owns everything that touches the world (the sandbox reset, the client, the telemetry files), so you can reuse the loop in your own code:

```python
from agent import run_agent, make_client, SYSTEM, TOOLS
```

**`system_prompt.md`.** Each episode's system prompt is a markdown file next to `agent.py`, loaded in one line. The prompt shares a common core across every episode; later episodes add only the section for the mechanism they introduce (Skills adds a skills section, for example). Diff two of them to see exactly what an episode taught the agent.

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
python run.py --cwd episodes/03-loop            # into logs/<timestamp>/
python run.py --cwd episodes/03-loop --capture  # also the terminal output
```

Each run gets its own timestamped folder under the episode's `logs/`, so you can run the same task repeatedly and compare how the agent's path and tool-call count vary from run to run. `capture.py` is the underlying terminal recorder and also works standalone on any command (e.g. `python capture.py -- pytest -q`).

## Seeing what the agent built

After a run, the agent's version of md2html is in the episode's `sandbox/`. To try it on a real document, render one of the sample files in `examples/` with it (from the repo root):

```bash
python render.py --cwd episodes/04-tools examples/emoji.md          # writes examples/emoji.html
python render.py --cwd episodes/04-tools examples/emoji.md --open   # and opens it in a browser
```

The page is written next to the Markdown file, as a complete page with md2html's built-in stylesheet. `render.py` runs md2html from inside the sandbox, so it always uses the agent's copy; an `md2html` command on your PATH would run whichever copy pip installed. Without `--cwd`, it uses the finished md2html in `examples/md2html/`.

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
