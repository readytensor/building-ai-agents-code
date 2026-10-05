"""
Episode 2 — Tools

Adds general primitives (list_files, read, write, edit, grep) alongside bash,
plus a tiny @tool decorator that builds each tool's JSON-schema from its
signature.
The tools now live in tools.py; this file is just the agent loop — which is
identical to Ep 1 except for dispatching by tool name. Completion is still
the natural stop: the loop ends when the model emits no tool calls.

See ../../README.md for context.
"""
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tiktoken import get_encoding

import tools as tools_module  # aliased: run_agent's `tools` parameter takes the canonical name; the loop sets tools_module.CURRENT_ROUND each turn
from tools import SANDBOX, TOOLS, write_tool_telemetry

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The agent's working directory: a fresh copy of initial/, reset by main().
# SANDBOX itself is defined in tools.py — the tools are bound to it.
INITIAL = Path("initial")


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


# The system prompt lives in system_prompt.md next to this file: prompt text is
# configuration, not loop logic. Its core is shared verbatim by every episode.
SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")
# The task: write a README. This continues Ep 1's second task -- only now the
# agent has real file tools, so the file lands in a single write() call instead
# of the many shell-escaping workarounds the bash-only agent needed in Ep 1.
TASK = (
    "This project has no README. Explore the codebase in the current directory "
    "and write a README.md for it. Cover: what the project does, how to install "
    "and use it (including the CLI), its architecture, and how to run the tests. "
    "Base everything on what you actually find in the code; don't guess."
)

# --- Usage telemetry: token counts per run, recorded by run_agent as it goes.
# The agent only RECORDS (to metrics.json); the harness (run.py) RENDERS the
# summary. (Tool-call telemetry lives in tools.py, next to the decorator that
# records it.)
USAGE = {
    "iterations": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "per_iter": [],  # {model_in, model_out, tools, tools_out} per round
}


def write_metrics(model: str, system: str, task: str):
    """Write this run's token usage to metrics.json. Recording only — the
    harness (run.py) reads this and renders the summary."""
    metrics = {
        "agents": [{"label": "agent", **USAGE}],
        "inputs": {"system": system, "task": task},
        "config": {"MODEL": model},
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# tiktoken encoder for the per-round tool-result token count (tools_out). Most of
# the context growth is tool results (a file read dwarfs the model's request), so
# we measure their real token size. cl100k_base is OpenAI's tokenizer; on Claude
# it's a close approximation — fine for a telemetry count.
_TOKENIZER = get_encoding("cl100k_base")


def _count_tokens(messages):
    """Real token count (tiktoken) of these messages' content — used to record
    each round's tool-result total (tools_out)."""
    return len(_TOKENIZER.encode("\n".join(str(m.get("content") or "") for m in messages)))


# --- The agent loop, as a function. The signature is the anatomy of an agent:
# a model, a system prompt, tools, and a task — give it those, get the final
# answer. Identical to Ep 1 except `tools` is now a list of @tool-decorated
# functions (schemas AND dispatch derive from it) instead of one hardwired tool.
def run_agent(client, model: str, system: str, tools: list, task: str) -> str:
    """Run the agent loop on `task` until the model stops requesting tool
    calls; return its final message. Records token usage into USAGE along the
    way (tool calls record themselves in tools.py)."""
    tools_by_name = {t.__name__: t for t in tools}
    tool_defs = [t.tool_definition for t in tools]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    iteration = 0

    while True:
        iteration += 1
        tools_module.CURRENT_ROUND = iteration   # tag tool calls with the round they happen in
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tool_defs,
        )
        usage = resp.usage
        USAGE["iterations"] = iteration
        USAGE["input_tokens"] += usage.prompt_tokens
        USAGE["output_tokens"] += usage.completion_tokens
        USAGE["per_iter"].append({"model_in": usage.prompt_tokens, "model_out": usage.completion_tokens, "tools": 0, "tools_out": 0})

        msg = resp.choices[0].message
        USAGE["per_iter"][-1]["tools"] = len(msg.tool_calls or [])   # tool calls requested this round
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
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
                # Tool errors come back to the model as the tool result, not as an agent crash.
                # The model can self-correct on the next iteration.
                result = f"Error executing {tc.function.name}: {type(e).__name__}: {e}"
                print(f"  ! {result}")
            preview = result if len(result) < 5000 else result[:5000] + "...[truncated]"
            print(f"  {preview}\n")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            round_tool_msgs.append(tool_msg)
            messages.append(tool_msg)

        # Tool results are most of the context growth (a file read dwarfs the model's
        # request); record this round's tool-result tokens so the per-iter numbers add up.
        USAGE["per_iter"][-1]["tools_out"] = _count_tokens(round_tool_msgs)


# --- Setup and run. Everything with side effects lives here, so importing
# this module (to reuse run_agent or the tools) touches nothing.
def main():
    # Sandbox reset: every run starts from a clean copy of initial/.
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    shutil.copytree(INITIAL, SANDBOX)

    # LLM client. Which provider/model to use is runtime config, read from
    # .env; make_client (defined above) does the connecting.
    load_dotenv(Path("../../.env"))
    base_url = os.environ.get("LLM_BASE_URL") or ""
    model = os.environ.get("LLM_AGENT_MODEL", "deepseek/deepseek-v4-flash")
    client = make_client(base_url)

    print(f"USER: {TASK}\n")
    final = run_agent(client, model, SYSTEM, TOOLS, TASK)
    print(f"\n=== FINAL RESPONSE ===\n\n{final}")
    write_tool_telemetry()
    write_metrics(model, SYSTEM, TASK)


if __name__ == "__main__":
    main()
