"""
Episode 6 — Subagents

Adds a `delegate` tool + worker runtime to Ep 5. The orchestrator (the
top-level agent) has no codebase-mutation tools; it can only `delegate` to
worker agents who do the actual work. Multiple `delegate` calls in one
assistant turn run CONCURRENTLY (ThreadPoolExecutor).

What Ep 6 adds:

1. run_agent(task, agent_type) -> str — Ep 5's loop, refactored into a
   function and used recursively: the orchestrator is run_agent(TASK,
   "orchestrator"); each worker is run_agent(subtask, agent_type). ALL
   formerly-module-level state (plan, loaded skills, tools, history + folds,
   counters) is now per-call function-local — so concurrent workers don't
   share state. (This is why Ep 6's planning.py and skills.py expose per-call
   factories instead of module globals.) Compaction stays non-destructive
   per agent: each keeps its own append-only history and derives its own view
   (build_view). The one shared thing is messages.jsonl, where every agent
   appends its entries tagged with its label, so the whole tree lands in one
   file as it runs.

2. delegate(task, agent_type) — calls run_agent recursively. Bound only into
   the orchestrator's toolset, never a worker's.

3. Parallel dispatcher — when an assistant turn contains multiple `delegate`
   calls, they fan out through a ThreadPoolExecutor and their results come
   back as multiple tool messages. Mirrors how Anthropic ships subagents
   (Claude Code / Agent SDK): multiple Agent tool calls in one turn run
   concurrently.

4. .agents/<name>.md worker configs — frontmatter (name, description, tools
   allowlist, skills to pre-load) + body (the worker's system prompt). Parsed
   with the same parser as .skills/<name>/SKILL.md (skills.parse_frontmatter).
   Two ship in .agents/ (at the episode root): implementer + verifier.

Completion is the natural stop — a worker (or the orchestrator) finishes when
it produces an assistant turn with no tool calls; run_agent returns that final
text. There is no done tool. Rigorous completion is structural here: the
orchestrator dispatches a read-only `verifier` worker and only finishes once
the verifier reports a clean pass (verifier-owns-completion).

agent.py owns the LLM client, the run_agent loop, delegate, the parallel
dispatcher, and the .agents registry. The carried-forward primitives live in
tools.py / compaction.py / planning.py / skills.py (one-way imports).

Unlike Eps 1-5, the LLM client stays module-level: delegate's model-facing
signature can't thread a client through the run_agent recursion.

See ../../README.md for context.
"""
import itertools
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env before importing the local modules below — compaction.py reads its
# knobs from the environment at import time.
load_dotenv(Path("../../.env"))

import planning  # noqa: E402
import skills  # noqa: E402
import tools  # noqa: E402  module ref so run_agent can append to tools.TOOL_CALLS
from tools import SANDBOX, tool, write_tool_telemetry  # noqa: E402
from compaction import (  # noqa: E402
    COMPACTION_THRESHOLD, KEEP_LAST_ITERATIONS, build_view, compact)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- 1. The agent's working directory: a fresh copy of initial/, reset by
# main(). SANDBOX is defined in tools.py — the tools are bound to it.
INITIAL = Path("initial")

# --- 2. LLM client (same provider-portable setup as Eps 1-5).
def make_client(base_url: str) -> OpenAI:
    """Connect to the LLM provider behind `base_url` — any OpenAI-compatible
    endpoint. The matching API key is picked from the environment by provider,
    so switching providers means changing only LLM_BASE_URL, never moving keys
    around. Anything OpenAI-compatible (Together, DeepSeek, OpenRouter, …)
    falls through to OPENAI_API_KEY."""
    by_provider = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "googleapis": "GOOGLE_API_KEY",
        "manus": "MANUS_API_KEY",
    }
    key_var = "OPENAI_API_KEY"
    for fragment, provider_key_var in by_provider.items():
        if fragment in base_url:
            key_var = provider_key_var
            break
    return OpenAI(api_key=os.environ.get(key_var), base_url=base_url or None)


BASE_URL = os.environ.get("LLM_BASE_URL") or ""
MODEL = os.environ.get("LLM_AGENT_MODEL", "deepseek/deepseek-v4-flash")
client = make_client(BASE_URL)

SUMMARIZER_BASE_URL = os.environ.get("LLM_SUMMARIZER_BASE_URL") or BASE_URL
SUMMARIZER_MODEL = os.environ.get("LLM_SUMMARIZER_MODEL") or MODEL
summarizer_client = (
    client if SUMMARIZER_BASE_URL == BASE_URL
    else make_client(SUMMARIZER_BASE_URL)
)

# --- Knobs. (Compaction knobs live in compaction.py.)
# Orchestrator cap.
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 200))
MAX_WORKER_ITER = int(os.environ.get("MAX_WORKER_ITER", 60))   # per-worker cap

# Always-available, stateless file tools, by name. The plan/skills tools are
# per-call closures (bound in run_agent); delegate is bound only for the
# orchestrator.
STATELESS_TOOLS = dict(tools.TOOL_FUNCTIONS)

# --- 3. Thread-safe printing with a per-worker label, so the parallel
# transcript stays readable ('[orch]', '[w1-implementer]', …).
_PRINT_LOCK = threading.Lock()
_WORKER_COUNTER = itertools.count(1)


def _print(label: str, text: str) -> None:
    with _PRINT_LOCK:
        for line in text.splitlines() or [""]:
            print(f"[{label}] {line}", flush=True)


# The canonical transcript, written as the run goes rather than at the end — so
# a run that crashes mid-task still leaves everything up to that point on disk.
# It holds two kinds of entry: every message exactly as it happened, and one
# record per compaction. Every agent writes to the same file and tags its
# entries with its label (same shape as tool_calls.jsonl), so one file holds the
# whole tree; filter by "agent" to replay a single worker. Concurrent workers
# append from their own threads, hence the lock. Truncated once per run in
# main(), never in run_agent — a worker starting must not wipe its siblings.
MESSAGE_LOG = Path("messages.jsonl")
_LOG_LOCK = threading.Lock()


def log_entry(label: str, entry: dict) -> None:
    """Append one labelled entry to MESSAGE_LOG. Opened and closed per write, so
    the line is on disk before the next thing happens."""
    with _LOG_LOCK:
        with open(MESSAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"agent": label, **entry}) + "\n")


def _preview_args(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        if len(repr(v)) < 60:
            parts.append(f"{k}={v!r}")
        else:
            parts.append(f"{k}=<{len(str(v))} chars>")
    return ", ".join(parts)


def _truncate(text: str, limit: int = 5000) -> str:
    return text if len(text) < limit else text[:limit] + "...[truncated]"


# --- 4. Worker config: .agents/<name>.md -> AgentConfig.
@dataclass(frozen=True)
class AgentConfig:
    name: str
    description: str
    tools: tuple           # allowlist into the available tools
    skills: tuple          # skills pre-loaded before the first turn
    prompt: str            # system prompt for this agent_type


# Worker configs live at the episode root (alongside agent.py / skills.py), NOT
# inside initial/ or the sandbox: they're agent infrastructure, not part of the
# toy codebase the workers edit. Keeping them out of SANDBOX also means a task
# that operates over the workspace (grep, list_files, git diff) never sweeps the
# agent's own configs into the task content.
_AGENTS_DIR = Path(".agents")


def _load_agent_configs() -> dict:
    out = {}
    if not _AGENTS_DIR.exists():
        return out
    for p in sorted(_AGENTS_DIR.glob("*.md")):
        fm, body = skills.parse_frontmatter(p)
        name = fm.get("name") or p.stem
        out[name] = AgentConfig(
            name=name,
            description=fm.get("description", ""),
            tools=tuple(fm.get("tools", [])),
            skills=tuple(fm.get("skills", [])),
            prompt=body,
        )
    return out


# --- 5. Per-worker metrics.
@dataclass
class WorkerMetrics:
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compactions: int = 0
    compact_in: int = 0
    compact_out: int = 0
    plan_writes: int = 0
    list_skills_calls: int = 0
    load_skill_calls: int = 0
    delegate_calls: int = 0
    loaded_skill_names: list = field(default_factory=list)


# One entry per agent (orchestrator + every worker), keyed by label.
GLOBAL_METRICS = {}
_METRICS_LOCK = threading.Lock()


def write_metrics():
    """Write per-worker usage to metrics.json — one entry per agent. Recording
    only; the harness (run.py) renders the per-worker + aggregate summary."""
    agents = []
    for label, m in GLOBAL_METRICS.items():
        agents.append({
            "label": label,
            "iterations": m.iterations,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "compactions": m.compactions,
            "compact_in": m.compact_in,
            "compact_out": m.compact_out,
            "reasoning": {"write_plan": m.plan_writes},
            "skills": {
                "list_skills": m.list_skills_calls,
                "load_skill": m.load_skill_calls,
                "loaded": m.loaded_skill_names,
            },
            "delegate_calls": m.delegate_calls,
        })
    metrics = {
        "agents": agents,
        "inputs": {"system": ORCHESTRATOR_SYSTEM, "task": TASK},
        "config": {
            "MODEL": MODEL,
            "SUMMARIZER_MODEL": SUMMARIZER_MODEL,
            "COMPACTION_THRESHOLD": COMPACTION_THRESHOLD,
            "KEEP_LAST_ITERATIONS": KEEP_LAST_ITERATIONS,
            "MAX_ITERATIONS": MAX_ITERATIONS,
            "MAX_WORKER_ITER": MAX_WORKER_ITER,
        },
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# --- 6. The orchestrator's system prompt.
ORCHESTRATOR_SYSTEM = """You are an **orchestrator**.

Your job is to decompose the user's task into focused subtasks and dispatch
them to specialized worker agents via the `delegate` tool. You yourself have
NO codebase-mutation tools — you cannot read, write, edit, grep, or run
shell commands. All work happens through workers.

## How to work

1. **Discover.** Call `list_skills` to see what skills exist (workers can
   load these).

2. **Plan.** Call `write_plan` to record your decomposition. In each step,
   note whether it is **PARALLELIZABLE** (independent — can run concurrently
   with peers) or **SEQUENTIAL** (depends on a prior step's output).

3. **Dispatch in parallel WHEREVER POSSIBLE.** This is the whole point of
   orchestration. The rules:

   - When you have ≥2 independent subtasks, you **MUST issue ALL their
     `delegate` tool calls in the SAME assistant turn.** The runtime will
     dispatch them concurrently across threads and return all results in
     the next turn.
   - **DO NOT** dispatch a sequence of one-delegate-per-turn for
     independent work. If you find yourself thinking *"let me dispatch
     the first feature, see how it goes, then the second"* — STOP. Batch
     them. Single-delegate turns are reserved for steps that DEPEND on
     a prior step's output (e.g., the verifier runs after the implementers;
     a fix-up worker reacts to a specific failure).
   - **DO NOT** dispatch a "recon" or "exploration" worker as a first step
     to map out the codebase for you. If the user's task mentions specific
     files or paths, include those paths in the worker task strings —
     workers have `read`/`grep`/`bash` and can investigate themselves.
     A recon-first pattern serialises what could be parallel and burns
     budget on context the implementer worker is going to re-read anyway.

4. **Always verify before you finish.** After the implementation workers
   finish, dispatch a `verifier` worker (in a separate turn — it depends on
   their outputs). Only finish once the verifier reports a clean pass.

## Worker task strings

Workers see ONLY the string you pass — no inheritance from your context,
no access to your plan, no access to other workers' outputs. Each task
string MUST be SELF-CONTAINED:
- What to build (specific, scoped to that worker only)
- Where the relevant files live (paths, fixture locations)
- Success criteria for that specific worker
- Any context from prior workers' outputs that this worker needs

## Counter-patterns to actively avoid

- ❌ Dispatching one worker, reading its result, then dispatching the next
  independent worker. (Use parallel batching instead.)
- ❌ Doing "recon" via a first worker before dispatching implementers.
  (Just include paths in the implementer task strings.)
- ❌ Finishing before the verifier confirms a clean pass.

## Finishing

There is no "done" tool. You finish by producing a turn with NO tool calls:
once the verifier reports a clean pass, stop calling `delegate` and write
your final summary as plain text. That summary is what the user sees — make
it a brief structured report of what was implemented and what verification
confirmed. Do not stop while any criterion is still unverified."""


# --- 7. delegate + the recursive agent loop. Both live in this one file, so
# delegate can call run_agent (and run_agent can bind delegate into a worker's
# toolset) with no circular import to untangle — the names resolve at call
# time, by which point both are defined.
@tool(
    "Spawn a worker agent to do `task`. `agent_type` is one of the values "
    "from .agents/ (typically 'implementer' or 'verifier'). The worker "
    "starts fresh (no inherited context) with the task string you pass, "
    "plus its configured tools and pre-loaded skills. Returns the worker's "
    "final summary. Multiple `delegate` calls in ONE assistant turn run "
    "CONCURRENTLY."
)
def delegate(task: str, agent_type: str) -> str:
    if agent_type not in AGENT_CONFIGS:
        return (f"Error: unknown agent_type '{agent_type}'. "
                "Available: "
                f"{sorted(k for k in AGENT_CONFIGS if k != 'orchestrator')}")
    return run_agent(task, agent_type)


def run_agent(task: str, agent_type: str) -> str:
    """The agent loop — used recursively. Each call owns its OWN per-call state
    (plan, loaded skills, tools registry, messages, metrics), so the
    orchestrator and any number of concurrent workers never share state.
    Returns the agent's final text (its natural-stop summary)."""
    cfg = AGENT_CONFIGS[agent_type]
    is_orchestrator = (agent_type == "orchestrator")
    label = "orch" if is_orchestrator else f"w{next(_WORKER_COUNTER)}-{agent_type}"

    def p(text):
        _print(label, text)

    # --- Per-call state.
    plan: list[dict] = []
    loaded_skills: dict = {}
    tools_by_name: dict = {}
    metrics = WorkerMetrics()
    with _METRICS_LOCK:
        GLOBAL_METRICS[label] = metrics

    # Bind this agent_type's allowlisted tools. File tools come straight from
    # STATELESS_TOOLS; plan/skills tools are per-call closures bound to the
    # state above; delegate is bound only for the orchestrator.
    for tname in cfg.tools:
        if tname in STATELESS_TOOLS:
            tools_by_name[tname] = STATELESS_TOOLS[tname]
        elif tname == "write_plan":
            tools_by_name["write_plan"] = planning.make_plan_tool(plan)
        elif tname == "list_skills":
            tools_by_name["list_skills"] = skills.make_list_skills_tool(loaded_skills)
        elif tname == "load_skill":
            tools_by_name["load_skill"] = skills.make_load_skill_tool(
                loaded_skills, tools_by_name)
        elif tname == "delegate" and is_orchestrator:
            tools_by_name["delegate"] = delegate

    # Pre-load any skills this agent_type requests in its config.
    for skill_name in cfg.skills:
        skill = skills._load_skill_body(skill_name)
        loaded_skills[skill_name] = skill
        for st in skill["tools"]:
            if st in skills._SKILL_TOOLS_REGISTRY:
                tools_by_name[st] = skills._SKILL_TOOLS_REGISTRY[st]
        metrics.loaded_skill_names.append(skill_name)

    # This agent's canonical history: append-only, and per-call like everything
    # else here — the orchestrator and each worker keep their own. Nothing is
    # ever removed from it; compaction records a fold instead, and what the
    # model sees is derived from the two.
    history: list[dict] = []
    folds: list[dict] = []

    def record(message):
        """Add a message to this agent's history, and to the shared on-disk log."""
        history.append(message)
        log_entry(label, {"kind": "message", **message})

    # Rebuilt each turn (plan+skills).
    record({"role": "system", "content": cfg.prompt})
    record({"role": "user", "content": task})
    iter_cap = MAX_ITERATIONS if is_orchestrator else MAX_WORKER_ITER

    p(f"=== START agent_type={agent_type} iter_cap={iter_cap} ===")
    p(f"task: {task[:300]}{'...' if len(task) > 300 else ''}")

    while metrics.iterations < iter_cap:
        metrics.iterations += 1

        # What this agent sees this turn, derived rather than stored: its history
        # projected through its folds, then the system prompt rebuilt from the
        # stable base + its plan + loaded-skill bodies. All live in per-call
        # state, so they are re-injected fresh each turn without entering the
        # transcript — and because the view is a fresh list, writing to it can't
        # disturb the record underneath.
        view = build_view(history, folds)
        view[0] = {
            "role": "system",
            "content": skills.system_with_skills(
                planning.system_with_plan(cfg.prompt, plan), loaded_skills
            ),
        }
        # The toolset grows as skills load, so rebuild the schema list each turn.
        tool_defs = [fn.tool_definition for fn in tools_by_name.values()]

        resp = client.chat.completions.create(
            model=MODEL, messages=view, tools=tool_defs,
        )
        u = resp.usage
        metrics.input_tokens += u.prompt_tokens
        metrics.output_tokens += u.completion_tokens

        msg = resp.choices[0].message
        record(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Natural stop — no tool calls means this agent is done. Its final
            # text is the result (the orchestrator's answer, or a worker's
            # summary handed back to whoever delegated it).
            text = msg.content or ""
            p(f"\n=== FINAL (natural stop) ===\n\n{text}")
            return text

        # Split delegate (parallelizable) from everything else (sequential).
        delegate_calls = [tc for tc in msg.tool_calls
                          if tc.function.name == "delegate"]
        other_calls = [tc for tc in msg.tool_calls if tc.function.name != "delegate"]
        results_by_id: dict = {}

        # Sequential dispatch for non-delegate tools.
        for tc in other_calls:
            try:
                fn = tools_by_name[tc.function.name]
                args = json.loads(tc.function.arguments)
                tools.TOOL_CALLS.append({"round": metrics.iterations, "agent": label,
                                         "tool": tc.function.name, "args": args})
                p(f"> {tc.function.name}({_preview_args(args)})")
                result = fn(**args)
                if tc.function.name == "write_plan":
                    metrics.plan_writes += 1
                elif tc.function.name == "list_skills":
                    metrics.list_skills_calls += 1
                elif tc.function.name == "load_skill":
                    metrics.load_skill_calls += 1
            except (TypeError, KeyError, json.JSONDecodeError, ValueError) as e:
                result = (f"Error executing {tc.function.name}: "
                          f"{type(e).__name__}: {e}")
                p(f"  ! {result}")
            p(f"  {_truncate(result)}")
            results_by_id[tc.id] = result

        # Delegate dispatch: a single call runs inline; ≥2 fan out concurrently.
        if delegate_calls:
            metrics.delegate_calls += len(delegate_calls)
            if len(delegate_calls) == 1:
                tc = delegate_calls[0]
                args = json.loads(tc.function.arguments)
                tools.TOOL_CALLS.append({"round": metrics.iterations, "agent": label,
                                         "tool": "delegate", "args": args})
                p(f"> delegate(agent_type={args.get('agent_type')!r}, "
                  f"task=<{len(str(args.get('task', '')))}chars>)")
                result = delegate(**args)
                p(f"  {_truncate(result)}")
                results_by_id[tc.id] = result
            else:
                types = [json.loads(tc.function.arguments).get("agent_type")
                         for tc in delegate_calls]
                p(f">>> Dispatching {len(delegate_calls)} workers in PARALLEL: "
                  f"{types}")
                with ThreadPoolExecutor(max_workers=len(delegate_calls)) as pool:
                    futures = {}
                    for tc in delegate_calls:
                        args = json.loads(tc.function.arguments)
                        tools.TOOL_CALLS.append({"round": metrics.iterations,
                                                 "agent": label, "tool": "delegate",
                                                 "args": args})
                        p(">    [submit] "
                          f"delegate(agent_type={args.get('agent_type')!r}, "
                          f"task=<{len(str(args.get('task', '')))}chars>)")
                        futures[pool.submit(delegate, **args)] = tc
                    for fut in as_completed(futures):
                        tc = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as e:
                            result = ("Error in worker delegate: "
                                      f"{type(e).__name__}: {e}")
                        atype = json.loads(tc.function.arguments).get("agent_type")
                        p(f">    [done] delegate({atype!r}): {_truncate(result)}")
                        results_by_id[tc.id] = result
                p(f">>> All {len(delegate_calls)} parallel workers complete.")

        # Append one tool message per tool_call_id, in call order (the API
        # requires a result for every tool call the assistant made).
        for tc in msg.tool_calls:
            record({"role": "tool", "tool_call_id": tc.id,
                    "content": results_by_id.get(tc.id, "")})

        # Compaction — per-worker, on this agent's own history. compact() reads
        # the current view and returns a summary; it never rewrites the history.
        view = build_view(history, folds)
        summary_msg, tail_len, ci, co, _middle = compact(
            view, summarizer_client, SUMMARIZER_MODEL)
        # Counted even when the summary was rejected — a guard-skipped attempt
        # still spent these tokens.
        metrics.compact_in += ci
        metrics.compact_out += co
        if summary_msg:
            # The fold: from here on, this summary stands in for everything
            # between the head and the preserved tail. Recording where the tail
            # starts is enough to locate it, because the tail is always the
            # newest messages and history only ever grows.
            fold = {
                "kind": "compaction",
                "tail_start": len(history) - tail_len,
                "summary": summary_msg,
            }
            folds.append(fold)
            log_entry(label, fold)
            metrics.compactions += 1
            p(f"  [COMPACTION FIRED — {len(view)} messages → "
              f"{len(build_view(history, folds))}, "
              f"history still {len(history)}, summarizer in={ci} out={co}]")
        elif ci:
            # The guard in compact() rejected a truncated/empty summary reply.
            p(f"  [COMPACTION SKIPPED — summary reply hit its token cap; "
              f"keeping full history this round (summarizer in={ci} out={co})]")

    # Iteration cap reached without a natural stop.
    last_text = ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            last_text = m.get("content") or ""
            break
    return (f"[worker '{agent_type}' hit its iteration cap ({iter_cap}) "
            "without finishing]\n"
            f"Last assistant text: {last_text[:300]}")


# --- 8. Build the agent registry. Worker configs come from .agents/; the
# orchestrator is defined here (it's the entry point, not a delegatable type).
AGENT_CONFIGS = _load_agent_configs()
AGENT_CONFIGS["orchestrator"] = AgentConfig(
    name="orchestrator",
    description="(top-level orchestrator; not dispatchable via delegate)",
    tools=("list_skills", "write_plan", "delegate"),
    skills=(),
    prompt=ORCHESTRATOR_SYSTEM,
)


# --- 9. The task.
TASK = """I want to round out our GFM support with three more features:

  1. Strikethrough: ~~text~~ -> <del>text</del>
  2. Task lists: list items starting with `- [ ]` or `- [x]` render
     with a disabled <input type="checkbox"> prepended (checked for [x]).
  3. Autolinks: <https://example.com> -> <a href="https://example.com">\
https://example.com</a>

Add each as a new extension under md2html/extensions/ and register
each in md2html/extensions/__init__.py. There are test fixture pairs
at tests/fixtures/strikethrough.{md,html}, task_lists.{md,html}, and
autolinks.{md,html} — all three currently fail because the features
aren't implemented.

Make sure all existing tests still pass. Keep diffs minimal."""


# --- 10. Setup and run. The sandbox reset and the invocation live here, so
# importing this module (to reuse run_agent) doesn't reset the sandbox or
# kick off a run.
def main():
    # Sandbox reset: every run starts from a clean copy of initial/.
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    shutil.copytree(INITIAL, SANDBOX)
    # One shared transcript for the whole agent tree — cleared here, once per
    # run, because every agent appends to it and none of them owns it.
    MESSAGE_LOG.unlink(missing_ok=True)

    print(f"USER: {TASK}\n")
    try:
        final_summary = run_agent(TASK, "orchestrator")
    except Exception as e:
        print(f"\n!!! TOP-LEVEL EXCEPTION: {type(e).__name__}: {e}")
        final_summary = f"(crashed: {e})"

    # Final output + metrics (recorded for run.py to render).
    print("\n" + "=" * 70)
    print("=== FINAL ORCHESTRATOR SUMMARY ===")
    print("=" * 70)
    print(final_summary)

    write_tool_telemetry()
    write_metrics()


if __name__ == "__main__":
    main()
