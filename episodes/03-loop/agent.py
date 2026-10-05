"""
Episode 1 — The Loop

Minimal agent: a while-loop calling a single `bash` tool until the model stops
requesting tool calls. Naive stop condition.

The loop lives in run_agent() — a function you can import and call from your
own code (give it a task, get the final answer). main() owns everything that
touches the world: the sandbox reset, the client, and the telemetry files.

See ../../README.md for context.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tiktoken import get_encoding

# Windows: make sure stdout can render UTF-8 (LLM outputs often contain → ✓ etc.)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The agent's working directory: a fresh copy of initial/, reset by main().
INITIAL = Path("initial")
SANDBOX = Path("sandbox")


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


# --- 1. The one tool: bash, bounded to the sandbox directory.
def bash(command: str) -> str:
    """Execute a shell command in the working directory and return its output.

    Failures (non-zero exit) come back as tool output so the model can read
    the error and adapt; crashing the agent on non-zero exit would defeat that.

    cwd only sets the STARTING directory -- it is not a security boundary.
    shell=True gives the model a real shell that can cd anywhere and touch
    anything this process can. True isolation needs a container or an OS
    sandbox (the way real agents such as Claude Code do it); out of scope for
    this toy, so run it on code you trust or inside a throwaway VM/container.
    """
    proc = subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=SANDBOX, encoding="utf-8", errors="replace",
        start_new_session=(os.name != "nt"),  # POSIX: own group so we can kill the whole tree
    )
    try:
        output = proc.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)  # boilerplate: shell=True orphans grandchildren
        return (
            "Error: command timed out after 30s and was killed (whole process "
            "tree). Avoid long-running or interactive commands, watch for code "
            "that can loop forever, and scope file searches to the working directory."
        )
    output = (output or "").strip()
    if len(output) > 20_000:                 # cap transcript growth from chatty commands
        output = output[:20_000] + "\n...[truncated]"
    if proc.returncode:                      # surface failures so the model can adapt
        output += f"\n(exit code {proc.returncode})"
    return output or "(no output)"


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a timed-out shell AND its descendants (cross-platform boilerplate).

    With shell=True the real command runs as a child of the shell; killing
    only the shell can leave that child alive, still holding the output pipe
    -- which deadlocks the drain (an unbounded loop would then hang forever).
    taskkill /T on Windows and killpg on POSIX take descendants down too.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    if proc.poll() is None:   # if the tree-kill missed, at least kill the shell
        proc.kill()

    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a shell command in the working directory and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
            },
            "required": ["command"],
        },
    },
}

# The tool list main() hands to run_agent — Ep 1 has exactly one tool. (Here
# this is the raw schema list the API sees; from Ep 2 on, TOOLS holds
# @tool-decorated functions and run_agent derives the schemas.)
TOOLS = [BASH_TOOL]

# The system prompt lives in system_prompt.md next to this file: prompt text is
# configuration, not loop logic. Its core is shared verbatim by every episode.
SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")
# Two tasks for the same agent. The agent is identical; only this string changes.
# The first asks it to summarize the project; the second asks it to write a README.
# To try the second, comment out the first TASK and uncomment the second.
TASK = (  # summarize the project
    "This project has no README. Explore the codebase in the current directory "
    "and tell me what it does: its purpose, how to use it, and how it's "
    "structured, in 100-150 words. Base it on what you actually find in the "
    "code; don't guess."
)
# TASK = (  # write the README
#     "This project has no README. Explore the codebase in the current directory "
#     "and write a README.md for it. Cover: what the project does, how to install "
#     "and use it (including the CLI), its architecture, and how to run the tests. "
#     "Base everything on what you actually find in the code; don't guess."
# )

# --- Tool-call telemetry: record every tool the agent invokes, in order, so
# we can see the path it took and how many calls it made (this varies run to
# run). Summarized and written to tool_calls.jsonl at the end of the run.
TOOL_CALLS = []  # list of {"round": n, "tool": name, "args": {...}} in call order


def write_tool_telemetry():
    """Write the tool calls made this run to tool_calls.jsonl, one JSON object
    per line in call order. Recording only — rendering a summary is left to
    whatever reads the file."""
    with open("tool_calls.jsonl", "w", encoding="utf-8") as f:
        for call in TOOL_CALLS:
            f.write(json.dumps(call) + "\n")


# --- Usage telemetry: token counts per run, recorded by run_agent as it goes.
# The agent only RECORDS (to metrics.json); the harness (run.py) RENDERS the
# summary. Keeping reporting out of the agent keeps it minimal.
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


# --- 2. The agent loop, as a function. The signature is the anatomy of an
# agent: a model, a system prompt, tools, and a task — give it those, get the
# final answer. `tools` is the JSON-schema list sent to the API; with exactly
# one tool this episode, dispatch below is hardwired to bash() (Ep 2
# generalizes it to dispatch by name).
def run_agent(client, model: str, system: str, tools: list, task: str) -> str:
    """Run the agent loop on `task` until the model stops requesting tool
    calls; return its final message. Records tool calls and token usage into
    TOOL_CALLS / USAGE along the way."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    iteration = 0

    while True:
        iteration += 1
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
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
            args = json.loads(tc.function.arguments)
            TOOL_CALLS.append({"round": iteration, "tool": tc.function.name, "args": args})
            print(f"> bash({args['command']!r})")
            result = bash(**args)
            if len(result) < 5000:
                preview = result
            else:
                preview = result[:5000] + "...[truncated]"
            print(f"  {preview}\n")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            round_tool_msgs.append(tool_msg)
            messages.append(tool_msg)

        # Tool results are most of the context growth (a file read dwarfs the model's
        # request); record this round's tool-result tokens so the per-iter numbers add up.
        USAGE["per_iter"][-1]["tools_out"] = _count_tokens(round_tool_msgs)


# --- 3. Setup and run. Everything with side effects lives here, so importing
# this module (to reuse run_agent or bash) touches nothing.
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
