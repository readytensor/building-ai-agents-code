"""The reference agent under evaluation: the Episode 5 agent, parameterized.

Reuses Ep 5's modules (tools, compaction, planning, skills) unchanged by putting
the episode directory on sys.path, then adapts them for an arbitrary repo:

  - tools.SANDBOX is repointed at the instance's working copy (the file tools
    resolve every path against it), so the agent edits the real repo, not
    episodes/05-skills/sandbox. On container runs (SWE-bench) there is no host
    copy at all: bash AND the file tools execute inside the instance's own
    container against its /testbed checkout (see container.py).
  - skills._SKILLS_DIR is repointed at eval/skills (generalized for any repo;
    verification runs the repo's own tests rather than md2html-specific coverage).
  - Ep 5's top-level loop is reproduced here as solve(); the initial/->sandbox
    reset is dropped because the runner owns repo state.

This module talks to the LLM, so it is exercised only by an explicit smoke run,
never the automated test suite.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EP5 = _REPO_ROOT / "episodes" / "05-skills"
sys.path.insert(0, str(_EP5))

load_dotenv(_REPO_ROOT / ".env")

import skills  # noqa: E402
import tools  # noqa: E402
from tools import tool, write_tool_telemetry  # noqa: E402
from compaction import build_view, compact, _count_tokens  # noqa: E402
from planning import write_plan, system_with_plan  # noqa: E402
from skills import list_skills, load_skill, system_with_skills  # noqa: E402

from eval import container, container_fileops  # noqa: E402

# Point the reused modules at eval's own skills (generalized for arbitrary
# repos). The instance repo is pointed at per call, in solve().
skills._SKILLS_DIR = _REPO_ROOT / "eval" / "skills"

# Eval runs have their own model config so the shared LLM_* vars can keep
# driving the episodes. Precedence: EVAL_LLM_* (from .env or inline) falls back
# to the episodes' LLM_* if unset.
MODEL = os.environ.get("EVAL_LLM_AGENT_MODEL") or os.environ.get("LLM_AGENT_MODEL", "deepseek/deepseek-v4-flash")
BASE_URL = os.environ.get("EVAL_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or ""
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 200))

# Advisor (optional): a stronger model the agent can consult mid-task -- the
# "executor + advisor" pattern. The ask_advisor tool registers ONLY when
# LLM_ADVISOR_MODEL is set, so a run without it is byte-identical to the
# plain agent. The advisor sees the executor's full live transcript and
# answers with bounded advice (never edits).
ADVISOR_MODEL = os.environ.get("LLM_ADVISOR_MODEL") or ""
ADVISOR_BASE_URL = os.environ.get("LLM_ADVISOR_BASE_URL") or BASE_URL
# Runaway cap on the advisor's reply, not a budget. Reasoning models (Sonnet 5
# etc.) spend "thinking" tokens against this same limit BEFORE any visible
# text, and the thinking length is unpredictable -- 2000 starved the reply to
# empty in testing, and so did 8000 later. Sized like the summarizer's cap:
# generous enough that a healthy call never hits it.
ADVISOR_MAX_TOKENS = int(os.environ.get("ADVISOR_MAX_TOKENS", 50_000))
# What the advisor is shown, now that compaction is non-destructive and both
# are available:
#   "view" (default) -- exactly what the executor sees this turn, folds and
#     all. Its advice can never lean on context the executor has lost, and the
#     input stays bounded for free.
#   "full" -- the complete append-only history, which is the thing keeping it
#     buys: the advisor sees the detail a summary dropped, which is the point
#     of a second opinion. Costs an input that grows all run, so it is trimmed
#     to the newest ADVISOR_CONTEXT_MAX_TOKENS when it outgrows the advisor's
#     own context window.
ADVISOR_CONTEXT = os.environ.get("ADVISOR_CONTEXT", "view")
ADVISOR_CONTEXT_MAX_TOKENS = int(os.environ.get("ADVISOR_CONTEXT_MAX_TOKENS", 150_000))

# The canonical transcript, appended to as the run goes -- so a sample that dies
# mid-solve (disk exhaustion, an OOM'd worker, a killed batch) still leaves its
# transcript behind. transcript.json is the same content written at the end.
# Two entry kinds: "message" for everything that happened, "compaction" for each
# fold. Replay the messages for what happened, apply the folds for what the
# model saw.
MESSAGE_LOG = Path("messages.jsonl")


def log_entry(entry: dict) -> None:
    """Append one entry to MESSAGE_LOG, flushed by closing the file."""
    with open(MESSAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# The system prompt lives in eval/system_prompt.md, byte-identical to Ep 5's
# (the episode this agent is built from); a drift test keeps every copy in sync.
SYSTEM = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")

# Advisor steering, rung 2: description-only steering produced zero calls
# (the skills precedent repeating), so a configured advisor also adds a
# mechanical consult directive -- the named-tool mandate pattern that
# reliably fires. Appended at runtime, gated on the env var, so the shared
# system_prompt.md and its drift test stay untouched; no-advisor runs use
# the file byte-for-byte.
if ADVISOR_MODEL:
    SYSTEM += (
        "\n\n## Advisor\n"
        "A more capable model is available through the ask_advisor tool. You "
        "MUST consult it at two points. (1) BEFORE your first edit to any "
        "file: state the approach you intend to take and ask whether it is "
        "the right one. (2) AFTER your changes pass verification, BEFORE you "
        "write your final summary: ask whether anything is wrong or missing. "
        "Weigh its advice against the code; you make the final call."
    )


# Ep 5's bash runs on the host. For SWE-bench instances the runner starts the
# instance's own Docker container (its real interpreter + frozen deps) and this
# proxy sends commands there instead -- so the agent can actually run the
# repo's test suite and get feedback on its edits. With no container active
# (e.g. the local md2html provider), it falls through to the host bash.
# tools.bash is the @tool-wrapped version; __wrapped__ is the original function
# (calling the wrapped one here would record each call in the telemetry twice).
_host_bash = tools.bash.__wrapped__


@tool("Execute a shell command in the repository's own environment and return "
      "its output. Use this to explore the project and to run its test suite "
      "to verify your changes.")
def bash(command: str) -> str:
    if container.ACTIVE:
        return container.exec_bash(container.ACTIVE, command)
    return _host_bash(command)


# On a container run the agent's whole workspace is the container's own
# /testbed checkout, so the file tools execute inside it too (identical
# semantics, via container_fileops piped over docker exec). bash routes itself
# in the proxy above; on local runs everything falls through to Ep 5's host
# implementations. The model sees the same tool names and schemas either way.
_IN_CONTAINER_TOOLS = {"list_files", "read", "write", "edit", "grep"}


# Routes itself like bash: the _IN_CONTAINER_TOOLS set is only for episode
# tools whose host implementations are container-unaware; this one checks
# container.ACTIVE in its own body.
@tool("Show a generated map of the repository. With no path (or '.'): the "
      "README and config files that exist, the package tree with docstring "
      "one-liners, and where the tests live. With path=<subdirectory>: the "
      "classes and function signatures in that subtree. Use it to decide "
      "where to read and edit instead of guessing.")
def repo_map(path: str = ".") -> str:
    if container.ACTIVE:
        return container.fileop(container.ACTIVE, "repo_map", {"path": path})
    return container_fileops.repo_map(tools.SANDBOX, path)



# The advisor reads the executor's LIVE conversation. It holds the history and
# folds themselves rather than a message list: both are stable objects that are
# only ever appended to, so solve() sets these once and they can't go stale.
# (Before compaction became non-destructive, compact() returned a NEW list each
# time and this reference had to be refreshed every iteration.)
_ADVISOR_STATE = {"history": None, "folds": None, "client": None, "in": 0, "out": 0}

_ADVISOR_SYSTEM = (
    "You are a senior engineer advising a colleague who is mid-task on a "
    "software issue. You are given their full working transcript and a "
    "question. Give focused, concrete advice: name the files, functions, and "
    "layers that matter, call out wrong assumptions or a wrong layer choice, "
    "and recommend a course of action. If more than one approach is on the "
    "table, say which one the issue wording and the codebase's own idioms "
    "favor, and why. Advise -- do not write the full patch."
)


def _advisor_messages() -> list:
    """The messages the advisor is shown, per ADVISOR_CONTEXT (see above)."""
    history = _ADVISOR_STATE["history"] or []
    folds = _ADVISOR_STATE["folds"] or []
    if ADVISOR_CONTEXT != "full":
        return build_view(history, folds)
    if _count_tokens(history) <= ADVISOR_CONTEXT_MAX_TOKENS:
        return history
    # Too big for one call: keep the head (system prompt + the task, which the
    # advice has to be aimed at) and as many of the NEWEST messages as fit.
    # A blunt cut on purpose -- summarizing here would just rebuild the view
    # that "full" was chosen to avoid. Cutting mid-conversation can orphan a
    # tool result from its call, which is fine here and only here: this list is
    # rendered to text for a single prompt, never sent as a message array.
    head, rest = history[:2], history[2:]
    budget = ADVISOR_CONTEXT_MAX_TOKENS - _count_tokens(head)
    kept = []
    for m in reversed(rest):
        budget -= _count_tokens([m])
        if budget <= 0:
            break
        kept.append(m)
    return head + list(reversed(kept))


def _render_transcript(messages) -> str:
    """The executor's conversation rendered for the advisor: system prompt and
    task verbatim, assistant turns with their tool calls inlined, tool results
    as-is (they are already strings, and compaction has already bounded the
    total size)."""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in m["tool_calls"]
            )
            content = (content + "\n" if content else "") + f"[tool calls: {calls}]"
        lines.append(f"--- {role} ---\n{content}")
    return "\n\n".join(lines)


@tool("Ask a more capable AI model for guidance on this task. It is "
      "automatically given your full conversation so far -- everything you "
      "have read, tried, and concluded -- so pass only your specific "
      "question. It returns concrete advice: which files or layers to "
      "change, which of several candidate approaches to take, what you may "
      "have missed. Consult it before committing to an approach, when you "
      "are stuck or torn between options, and before finishing. It cannot "
      "edit files or run commands; you act on its advice.")
def ask_advisor(question: str) -> str:
    prompt = (
        "Transcript of my work so far:\n\n"
        + _render_transcript(_advisor_messages())
        + "\n\n=== MY QUESTION ===\n" + question
    )
    resp = _chat_with_retry(
        _ADVISOR_STATE["client"], model=ADVISOR_MODEL,
        messages=[{"role": "system", "content": _ADVISOR_SYSTEM},
                  {"role": "user", "content": prompt}],
        max_tokens=ADVISOR_MAX_TOKENS,
    )
    if resp.usage is not None:
        _ADVISOR_STATE["in"] += resp.usage.prompt_tokens
        _ADVISOR_STATE["out"] += resp.usage.completion_tokens
    return resp.choices[0].message.content or "(the advisor returned no advice)"


def _excerpt(text: str, edge: int = 300) -> str:
    """Head AND tail of a long output: the head holds the session header or
    first error, the tail holds the verdict (pytest totals, the traceback's
    exception line, the appended exit code). Short outputs stay verbatim."""
    if len(text) <= 2 * edge:
        return text
    return text[:edge] + "\n...[snip]...\n" + text[-edge:]


def _call_tool(by_name, name, args):
    if container.ACTIVE and name in _IN_CONTAINER_TOOLS:
        # Same telemetry record the @tool wrapper writes on the host path.
        tools.TOOL_CALLS.append({"round": tools.CURRENT_ROUND, "tool": name, "args": dict(args)})
        result = container.fileop(container.ACTIVE, name, args)
    else:
        result = by_name[name](**args)
    # Postmortems need what the tool RETURNED, not just what was asked (a miss
    # is often explained by the output the model saw). Every tool is @tool-
    # wrapped, so the record this call appended is the last one; tag it.
    if tools.TOOL_CALLS:
        tools.TOOL_CALLS[-1]["result_excerpt"] = _excerpt(str(result))
    return result


def _client(base_url):
    def api_key_for(url):
        by_provider = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                       "groq": "GROQ_API_KEY", "googleapis": "GOOGLE_API_KEY", "manus": "MANUS_API_KEY"}
        for fragment, key_var in by_provider.items():
            if fragment in url:
                return os.environ.get(key_var)
        return os.environ.get("OPENAI_API_KEY")
    # Bounded waits, learned the hard way: the SDK's defaults (600s timeout,
    # silent retries) let a stalled provider wedge a worker for 30+ minutes
    # with no output. 120s per attempt x 3 attempts caps the silence at ~6
    # minutes; the worker then dies loudly and the dispatcher records it.
    return OpenAI(api_key=api_key_for(base_url), base_url=base_url or None,
                  timeout=120.0, max_retries=2)


# 1 original attempt + 2 retries, 20s apart. Roughly one call in every ~45
# samples' worth of traffic comes back as a 200 whose body is not JSON (seen
# live on django-15268: JSONDecodeError killed the worker 30 iterations in);
# the SDK retries HTTP-level failures, but a garbage 200 sails past that.
# Bounded on purpose -- a systematically broken provider (the Nemotron free-
# tier lesson) must still fail loudly, and the dispatcher records it.
LLM_ATTEMPTS = 3
LLM_RETRY_WAIT = 20  # seconds


def _chat_with_retry(client, sleep=time.sleep, **kwargs):
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except (json.JSONDecodeError, APIError) as e:
            if attempt == LLM_ATTEMPTS:
                raise
            print(f"[llm] {type(e).__name__} on attempt {attempt}/{LLM_ATTEMPTS}, "
                  f"retrying in {LLM_RETRY_WAIT}s", flush=True)
            sleep(LLM_RETRY_WAIT)


def solve(repo_dir: Path, problem_statement: str, audit=None) -> str:
    """Run the Ep 5 agent with problem_statement as the task. repo_dir is the
    host working copy to edit in place -- or None on a container run, where
    the workspace is the container's /testbed and every tool executes inside.
    Returns "" either way: the runner owns diff capture.

    audit is the environment's half of the stop handshake (see eval/audit.py):
    when the model requests a stop (a turn with no tool calls), the hook is
    called and its findings -- if any, and if the one-bounce budget is unspent
    -- go back into the conversation instead of ending the run. None (the
    episodes, the local provider) keeps the classic natural stop, verbatim."""
    # Reset per-run state. The episodes never reset TOOL_CALLS (one run per
    # process), but a multi-sample batch runs several solves in one process --
    # without the clear, each sample's tool_calls.jsonl would accumulate every
    # earlier sample's calls.
    skills.LOADED_SKILLS.clear()
    skills.LOADED_TOOLS.clear()
    tools.TOOL_CALLS.clear()
    if repo_dir is not None:  # container runs have no host working copy
        tools.SANDBOX = Path(repo_dir).resolve()

    client = _client(BASE_URL)
    # Ep 5's toolset, with its host-only bash swapped for the container-aware
    # proxy above (same name, same schema shape -- the model sees no difference).
    file_tools = [bash if t.__name__ == "bash" else t for t in tools.TOOLS]
    base_tools = file_tools + [repo_map, write_plan, list_skills, load_skill]
    if ADVISOR_MODEL:
        base_tools.append(ask_advisor)
    base_by_name = {t.__name__: t for t in base_tools}

    # Append-only canonical history + the folds compaction records against it;
    # what the model sees each turn is derived from the two (Ep 3's mechanism).
    history: list = []
    folds: list = []
    MESSAGE_LOG.unlink(missing_ok=True)

    def record(message):
        """Add a message to the canonical history, and to the on-disk log."""
        history.append(message)
        log_entry({"kind": "message", **message})

    record({"role": "system", "content": SYSTEM})
    record({"role": "user", "content": problem_statement})
    _ADVISOR_STATE.update({
        "history": history, "folds": folds,
        "client": client if ADVISOR_BASE_URL == BASE_URL else _client(ADVISOR_BASE_URL),
        "in": 0, "out": 0,
    })
    iteration = 0
    total_in = total_out = 0
    compactions = compact_in = compact_out = 0
    tool_call_count = 0
    mechanism_calls = {"write_plan": 0, "list_skills": 0, "load_skill": 0,
                       "repo_map": 0, "ask_advisor": 0}
    audit_bounces, audit_unresolved = 0, []
    while iteration < MAX_ITERATIONS:
        iteration += 1
        tools.CURRENT_ROUND = iteration
        # What the model sees this turn: derived from the history, then the
        # system prompt rebuilt with the plan and any loaded skills.
        view = build_view(history, folds)
        view[0] = {"role": "system",
                   "content": system_with_skills(system_with_plan(SYSTEM))}
        by_name = {**base_by_name, **skills.LOADED_TOOLS}
        tool_defs = [fn.tool_definition for fn in by_name.values()]

        resp = _chat_with_retry(client, model=MODEL, messages=view, tools=tool_defs)
        # Some providers (notably OpenRouter's free tiers) omit usage on some
        # responses; count what's reported rather than crashing the run.
        if resp.usage is not None:
            total_in += resp.usage.prompt_tokens
            total_out += resp.usage.completion_tokens
        msg = resp.choices[0].message
        record(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            # The stop handshake: the model just REQUESTED a stop; the audit
            # hook is the environment deciding whether to GRANT it. Findings
            # bounce back into the same conversation exactly once -- the agent
            # keeps its full context, so acting on them is cheap. After the
            # budget, the stop is unconditional (MAX_ITERATIONS backstops all).
            findings = audit() if audit else []
            if findings and audit_bounces < 1:
                audit_bounces += 1
                print(f"[iter {iteration}] [audit bounce: {len(findings)} finding(s)]",
                      flush=True)
                record({"role": "user", "content": (
                    "AUDIT: your submission was checked before acceptance and "
                    "was not accepted:\n"
                    + "\n".join(f"- {f}" for f in findings)
                    + "\nAddress these findings, then finish normally.")})
                continue
            audit_unresolved = findings  # accepted anyway: record what stood
            break
        tool_call_count += len(msg.tool_calls)
        for tc in msg.tool_calls:
            if tc.function.name in mechanism_calls:
                mechanism_calls[tc.function.name] += 1
            try:
                args = json.loads(tc.function.arguments)
                # Live progress: one short line per tool call, so a background
                # run can be followed with tail -f on its log.
                preview = ", ".join(
                    f"{k}={v!r}" if len(repr(v)) < 60 else f"{k}=<{len(str(v))} chars>"
                    for k, v in args.items()
                )
                print(f"[iter {iteration}] {tc.function.name}({preview})", flush=True)
                result = _call_tool(by_name, tc.function.name, args)
            except (TypeError, KeyError, json.JSONDecodeError, ValueError) as e:
                result = f"Error executing {tc.function.name}: {type(e).__name__}: {e}"
                print(f"[iter {iteration}] ! {result}", flush=True)
            record({"role": "tool", "tool_call_id": tc.id, "content": result})
        # compact() returns a summary and how much tail to keep; recording that
        # as a fold is what shrinks the next view. The history is never rewritten.
        summary_msg, tail_len, ci, co, _ = compact(build_view(history, folds), client, MODEL)
        if summary_msg:
            fold = {"kind": "compaction",
                    "tail_start": len(history) - tail_len,
                    "summary": summary_msg}
            folds.append(fold)
            log_entry(fold)
            compactions += 1
            compact_in += ci
            compact_out += co
            print(f"[iter {iteration}] [compaction fired: summarizer in={ci} out={co}]", flush=True)

    write_tool_telemetry()
    # The closing summary is the only place the agent states what it did and
    # what it could not verify -- without this file that conclusion is lost
    # (learned on sympy-21612: we couldn't tell an honest blocked engineer
    # from a confused agent).
    final = next((m.get("content") or "" for m in reversed(history)
                  if m.get("role") == "assistant"), "")
    with open("final_message.md", "w", encoding="utf-8") as f:
        f.write(final)
    # The full message history, verbatim: every model turn, every complete tool
    # result -- now the ORIGINAL history rather than the post-compaction state,
    # which a run with compactions used to lose entirely. tool_calls.jsonl
    # excerpts are the quick-scan layer; this is the replay-anything layer, and
    # messages.jsonl is the same content written as the run goes so it survives
    # a crash, plus the folds -- replay those to see what the model actually saw.
    with open("transcript.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    # Same recording split as the episodes: the agent writes raw counters, the
    # harness owns collection/reporting. The runner moves this into the batch dir.
    metrics = {
        "agents": [{
            "label": "agent",
            "iterations": iteration,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "tool_calls": tool_call_count,
            "compactions": compactions,
            "compact_in": compact_in,
            "compact_out": compact_out,
            "reasoning": {"write_plan": mechanism_calls["write_plan"]},
            "skills": {
                "list_skills": mechanism_calls["list_skills"],
                "load_skill": mechanism_calls["load_skill"],
                "loaded": list(skills.LOADED_SKILLS),
            },
            "orientation": {"repo_map": mechanism_calls["repo_map"]},
            "advisor": {
                "model": ADVISOR_MODEL or None,
                "calls": mechanism_calls["ask_advisor"],
                "advisor_in": _ADVISOR_STATE["in"],
                "advisor_out": _ADVISOR_STATE["out"],
            },
            "audit": {"bounces": audit_bounces, "unresolved": audit_unresolved},
        }],
        "config": {"MODEL": MODEL, "MAX_ITERATIONS": MAX_ITERATIONS},
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return ""  # the runner captures the diff from the repo's git state
