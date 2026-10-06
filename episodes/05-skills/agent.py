"""
Skills

Adds skills to the Tools agent: instructions the agent loads when it needs
them, instead of carrying every procedure in its system prompt on every call.
A skill is a SKILL.md with a name, a description and a body (skills.py). Every
skill's name and description go into the system prompt once, at startup; the
body arrives only when the agent calls load_skill, as a tool result. The loop
is the Tools loop unchanged: skills show up as one more tool and a longer
system prompt.

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

# Aliased: run_agent's `tools` parameter takes the canonical name; the loop sets
# tools_module.CURRENT_ROUND each turn.
import tools as tools_module
from skills import load_skill, skills_index
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
# The skills index closes the prompt: one line per skill, so the model knows what
# it can load without any skill's body in context yet.
SYSTEM += "\n## Available skills\n\n" + skills_index() + "\n"
# The task: a real feature for md2html, whose rules are the team's, not the
# model's: the style guide is a skill, and nothing in the repository states it.
TASK = """Our documents are typed with straight quotes, plain hyphens and three
dots, and our style guide asks for typographic punctuation in the rendered
HTML: curly quotes, proper dashes, the ellipsis character. Add a typography
extension to md2html that applies the team's style guide for punctuation to
the text of a document, exactly as the guide states it:

- Nothing changes inside code spans, code blocks, or link addresses.
- Implement it as a new extension under md2html/extensions/, registered like
  the existing ones.

Add tests for the rules, and make sure the existing tests still pass too."""

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
    text = "\n".join(str(m.get("content") or "") for m in messages)
    return len(_TOKENIZER.encode(text))


# --- The agent loop, as a function. The signature is the anatomy of an agent:
# a model, a system prompt, tools, and a task — give it those, get the final
# answer. The same as before except `tools` is now a list of @tool-decorated
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
        # tag tool calls with the round they happen in
        tools_module.CURRENT_ROUND = iteration
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tool_defs,
        )
        usage = resp.usage
        USAGE["iterations"] = iteration
        USAGE["input_tokens"] += usage.prompt_tokens
        USAGE["output_tokens"] += usage.completion_tokens
        USAGE["per_iter"].append({
            "model_in": usage.prompt_tokens, "model_out": usage.completion_tokens,
            "tools": 0, "tools_out": 0,
        })

        msg = resp.choices[0].message
        # tool calls requested this round
        USAGE["per_iter"][-1]["tools"] = len(msg.tool_calls or [])
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
                # Tool errors come back to the model as the tool result, not as an
                # agent crash. The model can self-correct on the next iteration.
                result = (
                    f"Error executing {tc.function.name}: {type(e).__name__}: {e}"
                )
                print(f"  ! {result}")
            if len(result) < 5000:
                preview = result
            else:
                preview = result[:5000] + "...[truncated]"
            print(f"  {preview}\n")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            round_tool_msgs.append(tool_msg)
            messages.append(tool_msg)

        # Tool results are most of the context growth (a file read dwarfs the
        # model's request); record this round's tool-result tokens so the
        # per-iter numbers add up.
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
    # The tools: the Tools episode's, plus load_skill.
    final = run_agent(client, model, SYSTEM, TOOLS + [load_skill], TASK)
    print(f"\n=== FINAL RESPONSE ===\n\n{final}")
    write_tool_telemetry()
    write_metrics(model, SYSTEM, TASK)


if __name__ == "__main__":
    main()
