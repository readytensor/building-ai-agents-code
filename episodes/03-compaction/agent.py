"""
Episode 3 — Compaction

Adds one thing to Ep 2's agent: rolling-summary compaction. When the compactable
*middle* of the message history grows past COMPACTION_THRESHOLD, that middle is
summarized and one summary message stands in for it — so a long-running task
doesn't keep re-paying for the full transcript on every turn.

The history itself is append-only. Nothing is ever deleted from it: compaction
records a *fold* (a summary plus where the preserved tail starts), and what we
send the model each turn is derived from the two by build_view(). Compaction
shrinks the payload, not the record — and the whole run, folds included, is
appended to messages.jsonl as it happens, so a run that dies mid-task still
leaves a complete transcript behind.

The action space is unchanged from Ep 2 (same tools.py). Completion is still the
natural stop: the loop ends when the model emits no tool calls — its trained
instinct that the task is done. (Rigorous, externally-verified completion —
running pre-written tests and only stopping when they pass — arrives later, as
the verification skill.)

This file is just the agent loop. The compaction mechanism gets its own
compaction.py; both it and tools.py import one-way (`agent → tools`,
`agent → compaction`) — agent.py owns the LLM client and passes it into
compact().

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
# reads its knobs (threshold, keep, summarizer model) from the environment at
# import time. This is the one side effect that can't wait for main().
load_dotenv(Path("../../.env"))

# Aliased: run_agent's `tools` parameter takes the canonical name; the loop sets
# tools_module.CURRENT_ROUND each turn.
import tools as tools_module  # noqa: E402
from tools import SANDBOX, TOOLS, write_tool_telemetry  # noqa: E402
from compaction import (  # noqa: E402
    COMPACTION_THRESHOLD, KEEP_LAST_ITERATIONS, build_view, compact, _count_tokens,
)

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

# The system prompt lives in system_prompt.md next to this file: prompt text is
# configuration, not loop logic. Its core is shared verbatim by every episode.
SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")
TASK = """Our markdown documents are getting long and hard to navigate. Add a
table-of-contents extension to md2html:

- Every rendered heading gets an HTML id derived from its text; duplicate
  heading texts get distinct ids (-1, -2, ...).
- A line consisting of exactly [TOC] becomes a labeled, nested list of links
  to every heading. Without the marker: anchors only, no TOC.
- Implement it as a new extension under md2html/extensions/, registered like
  the existing ones.

I've added a fixture pair at tests/fixtures/toc.md and tests/fixtures/toc.html
showing the expected output; it currently fails. Make it pass, add your own
tests, and make sure the existing tests still pass too."""

# --- Usage telemetry: token counts per run, recorded by run_agent as it goes.
# The agent only RECORDS (to metrics.json); the harness (run.py) RENDERS the
# summary. Compaction tokens are recorded separately so the harness can show
# the agent-vs-compaction split. (Tool-call telemetry lives in tools.py.)
USAGE = {
    "iterations": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "compactions": 0,
    "compact_in": 0,
    "compact_out": 0,
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
# a model, a system prompt, tools, and a task — plus this episode's addition,
# the summarizer that compaction runs on.
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
        # What the model sees this turn: derived from the history, not the
        # history itself. Until the first compaction the two are identical.
        view = build_view(history, folds)
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
