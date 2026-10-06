"""
Episode 4 — Working Memory

Gives the agent durable, self-maintained state that survives compaction.

The mechanism is a *dynamic system prompt*: each iteration the loop rebuilds the
system message as [stable base + current plan]. Because the plan lives in agent
state (not message history) it survives compaction untouched, and because it
rides in the system prompt the message prefix stays stable when the plan is
unchanged. write_plan (see planning.py) is the worked instance of that durable
slot — a structured plan the agent writes and updates as it works; the same
mechanism is what Ep 5 builds on to inject loaded-skill bodies.

Everything else is Ep 3 unchanged: the action space (tools.py), rolling-summary
compaction (compaction.py — non-destructive: the history is append-only, and what
the model sees each turn is derived from it by build_view, which is where the
plan gets layered in; the run is appended to messages.jsonl as it goes), the
sandbox reset, and the natural stop — the loop
ends when the model emits no tool calls. (No self-assessed done tool; rigorous,
externally-verified completion arrives later, as the verification skill.)

This file is just the agent loop. It owns the LLM client and passes it into
compact(); imports are one-way (`agent → tools`, `agent → compaction`,
`agent → planning`).

See ../../README.md for context.
"""
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env at import — before the local imports below, because compaction.py
# reads its knobs (threshold, keep) from the environment at import time. This
# is the one side effect that can't wait for main().
load_dotenv(Path("../../.env"))

# Aliased: run_agent's `tools` parameter takes the canonical name; the loop sets
# tools_module.CURRENT_ROUND each turn.
import tools as tools_module  # noqa: E402
from tools import SANDBOX, TOOLS as BASE_TOOLS, write_tool_telemetry  # noqa: E402
from compaction import (  # noqa: E402
    COMPACTION_THRESHOLD, KEEP_LAST_ITERATIONS, build_view, compact, _count_tokens,
)
from planning import write_plan, system_with_plan  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The agent's working directory: a fresh copy of initial/, reset by main().
# SANDBOX itself is defined in tools.py — the tools are bound to it.
INITIAL = Path("initial")

# The canonical transcript, written as the run goes rather than at the end — so
# a run that crashes mid-task still leaves everything up to that point on disk.
# It holds two kinds of entry: every message exactly as it happened, and one
# record per compaction. Both views are recoverable from the one file — what
# happened is the messages in order; what the model saw at any point is the
# messages with each fold applied.
MESSAGE_LOG = Path("messages.jsonl")


def log_entry(entry: dict):
    """Append one entry to MESSAGE_LOG. Opened and closed per write, so the line
    is on disk before the next thing happens."""
    with open(MESSAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


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


# --- Loop safety cap to prevent an infinite loop. (Compaction knobs live in
# compaction.py.)
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 150))

# --- Tool registry: Ep 3's six file primitives plus Ep 4's new tool,
# write_plan. main() passes this list into run_agent, which derives the
# schema list and dispatch map from it. (Tool-call telemetry lives in tools.py.)
TOOLS = BASE_TOOLS + [write_plan]

# The system prompt lives in system_prompt.md next to this file: prompt text is
# configuration, not loop logic. Its core is shared verbatim by every episode;
# this episode's copy adds the Working plan section (the mechanism built here).
SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")
TASK = """I want to add support for reference-style links to our markdown
library. They look like this:

    Here is a [link][myref] in text.

    [myref]: https://example.com "Optional title"

The link definitions (the `[id]: url "title"` lines) get collected from
the document, and inline `[text][id]` references resolve to <a> elements
using those URLs. The definition lines themselves should NOT appear in
the rendered output.

This touches a few parts of the pipeline, so plan the work first and
track your progress against it as you go.

I've added a test fixture at tests/fixtures/reference_style_links.md and
tests/fixtures/reference_style_links.html showing the expected behavior;
it currently fails. Make it pass, and make sure the existing tests still
pass too."""

# --- Usage telemetry: token counts per run, recorded by run_agent as it goes.
# The agent only RECORDS (to metrics.json); the harness (run.py) RENDERS the
# summary. Compaction tokens are recorded separately, as is the write_plan
# count, so the harness can show the agent-vs-compaction split and how often
# the new tool fired. (Tool-call telemetry lives in tools.py.)
USAGE = {
    "iterations": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "compactions": 0,
    "compact_in": 0,
    "compact_out": 0,
    # "reasoning" is the shared metrics key across Eps 4-6 + run.py —
    # a stable grouping for write_plan's call count.
    "reasoning": {"write_plan": 0},
    # {model_in, model_out, tools, tools_out, middle, compacted} per round
    "per_iter": [],
}


def write_metrics(model: str, summarizer_model: str, system: str, task: str):
    """Write this run's token usage to metrics.json. Recording only — the
    harness (run.py) reads this and renders the summary."""
    metrics = {
        "agents": [{"label": "agent", **USAGE}],
        "inputs": {"system": system, "task": task},
        "config": {
            "MODEL": model,
            "SUMMARIZER_MODEL": summarizer_model,
            "COMPACTION_THRESHOLD": COMPACTION_THRESHOLD,
            "KEEP_LAST_ITERATIONS": KEEP_LAST_ITERATIONS,
            "MAX_ITERATIONS": MAX_ITERATIONS,
        },
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# --- The agent loop, as a function. The signature is the anatomy of an agent:
# a model, a system prompt, tools, and a task — plus Ep 3's summarizer. `system`
# is the stable base; the loop layers the current plan onto it each turn.
def run_agent(client, model: str, system: str, tools: list,
              summarizer_client, summarizer_model: str, task: str):
    """Run the agent loop on `task` until the model stops requesting tool calls
    (the natural stop); return its final message. Returns None if the loop hits
    MAX_ITERATIONS first. Records token usage into USAGE along the way (tool
    calls record themselves in tools.py)."""
    tools_by_name = {t.__name__: t for t in tools}
    tool_defs = [t.tool_definition for t in tools]
    # The canonical history: append-only. Nothing is ever removed from it or
    # rewritten in it — compaction records a fold instead (see below), and what
    # the model sees is derived from the two.
    history = []
    folds = []
    MESSAGE_LOG.unlink(missing_ok=True)

    def record(message):
        """Add a message to the canonical history, and to the on-disk log."""
        history.append(message)
        log_entry({"kind": "message", **message})

    record({"role": "system", "content": system})
    record({"role": "user", "content": task})
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        # Tag tool calls with the round they happen in.
        tools_module.CURRENT_ROUND = iteration

        # What the model sees this turn, derived rather than stored: the history
        # projected through its folds, then the system prompt rebuilt from the
        # stable base plus the current plan. The plan lives in agent state
        # (planning.CURRENT_PLAN), so it is re-injected fresh each turn without
        # ever entering the message history — and because the view is a fresh
        # list, writing to it can't disturb the record underneath.
        view = build_view(history, folds)
        view[0] = {"role": "system", "content": system_with_plan(system)}

        resp = client.chat.completions.create(
            model=model, messages=view, tools=tool_defs,
        )
        u = resp.usage
        USAGE["iterations"] = iteration
        USAGE["input_tokens"] += u.prompt_tokens
        USAGE["output_tokens"] += u.completion_tokens
        USAGE["per_iter"].append({
            "model_in": u.prompt_tokens, "model_out": u.completion_tokens,
            "tools": 0, "tools_out": 0, "middle": 0, "compacted": False,
        })

        msg = resp.choices[0].message
        # Tool calls requested this round.
        USAGE["per_iter"][-1]["tools"] = len(msg.tool_calls or [])
        record(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Natural stop: no tool calls means the model considers the task done.
            return msg.content or ""

        round_tool_msgs = []
        for tc in msg.tool_calls:
            try:
                fn = tools_by_name[tc.function.name]
                args = json.loads(tc.function.arguments)
                parts = []
                for k, v in args.items():
                    if len(repr(v)) < 60:
                        parts.append(f"{k}={v!r}")
                    else:
                        parts.append(f"{k}=<{len(str(v))} chars>")
                arg_preview = ", ".join(parts)
                print(f"> {tc.function.name}({arg_preview})")
                result = fn(**args)
                if tc.function.name == "write_plan":
                    USAGE["reasoning"]["write_plan"] += 1
            except (TypeError, KeyError, json.JSONDecodeError, ValueError) as e:
                # Bad tool call (missing args, unknown tool, etc.) — feed the error
                # back to the model so it can self-correct rather than crashing.
                result = (
                    f"Error executing {tc.function.name}: {type(e).__name__}: {e}"
                )
                print(f"  ! {result}")
            preview = (
                result if len(result) < 5000 else result[:5000] + "...[truncated]"
            )
            print(f"  {preview}\n")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            round_tool_msgs.append(tool_msg)
            record(tool_msg)

        # Tool results are most of the context growth: the model only *requests* a
        # tool (small `out`), but the result it hands back can be huge (a file read).
        # Record this round's tool-result tokens so the per-iter numbers actually add
        # up.
        USAGE["per_iter"][-1]["tools_out"] = _count_tokens(round_tool_msgs)

        # Compaction: compact() summarizes the older middle once the MIDDLE's own
        # token count crosses the threshold, and no-ops otherwise — so it's safe to
        # call every turn; it only summarizes when there's enough stale middle to be
        # worth it (and with KEEP small, the fire drops the input hard). It reads
        # the current view and returns a summary; it never rewrites the history.
        view = build_view(history, folds)
        summary_msg, tail_len, ci, co, middle_tok = compact(
            view, summarizer_client, summarizer_model)
        # Compactable-middle size this turn (the sawtooth metric).
        USAGE["per_iter"][-1]["middle"] = middle_tok
        # Counted even when the summary was rejected — a guard-skipped attempt still
        # spent these tokens.
        USAGE["compact_in"] += ci
        USAGE["compact_out"] += co
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
            log_entry(fold)
            USAGE["compactions"] += 1
            # The middle crossed the threshold this iteration.
            USAGE["per_iter"][-1]["compacted"] = True
            print(f"  [COMPACTION FIRED — {len(view)} messages → "
                  f"{len(build_view(history, folds))}, "
                  f"history still {len(history)}, summarizer in={ci} out={co}]\n")
        elif ci:
            # The guard in compact() rejected a truncated/empty summary reply.
            print(f"  [COMPACTION SKIPPED — summary reply hit its token cap; "
                  f"keeping full history this round (summarizer in={ci} out={co})]\n")

    return None   # iteration cap reached without a natural stop


# --- Setup and run. Everything with side effects lives here (except the .env
# load above), so importing this module to reuse run_agent touches nothing.
def main():
    # Sandbox reset: every run starts from a clean copy of initial/.
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    shutil.copytree(INITIAL, SANDBOX)

    # LLM client. Which provider/model to use is runtime config, read from
    # .env; make_client (defined above) does the connecting.
    base_url = os.environ.get("LLM_BASE_URL") or ""
    model = os.environ.get("LLM_AGENT_MODEL", "deepseek/deepseek-v4-flash")
    client = make_client(base_url)

    # Compaction summarizes on its own model — and, if pointed at a different
    # provider, its own endpoint. Both default to the agent's, so leaving the
    # LLM_SUMMARIZER_* vars unset simply reuses the agent's client.
    summarizer_base_url = os.environ.get("LLM_SUMMARIZER_BASE_URL") or base_url
    summarizer_model = os.environ.get("LLM_SUMMARIZER_MODEL") or model
    summarizer_client = (
        client if summarizer_base_url == base_url
        else make_client(summarizer_base_url)
    )

    print(f"USER: {TASK}\n")
    final = run_agent(client, model, SYSTEM, TOOLS,
                      summarizer_client, summarizer_model, TASK)
    if final is None:
        print(f"\n=== MAX_ITERATIONS REACHED ({MAX_ITERATIONS}) — aborting ===")
    else:
        print(f"\n=== FINAL RESPONSE ===\n\n{final}")
    write_tool_telemetry()
    write_metrics(model, summarizer_model, SYSTEM, TASK)


if __name__ == "__main__":
    main()
