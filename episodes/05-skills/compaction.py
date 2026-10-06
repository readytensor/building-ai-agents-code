"""
Episode 5 — Skills (compaction)

Ep 3's headline mechanism, carried forward unchanged: rolling-summary
compaction. When the compactable *middle* of the message history grows past
COMPACTION_THRESHOLD, that middle is summarized via a second LLM call and
one summary message stands in for it — so a long-running task doesn't keep
paying for the full transcript every turn. If the summarizer call fails (an
empty or cap-truncated reply), nothing is installed — the round is skipped and
the next turn retries — so a bad summary can never stand in for the history it
was meant to compress.

Compaction is *non-destructive*: compact() never rewrites the history it is
handed. It returns the summary it wrote and how many recent messages to keep
verbatim, and the caller records that as a **fold**. The canonical history
stays append-only, and what the model sees is rebuilt from it each turn by
build_view() — so the payload shrinks while the record doesn't.

This file is identical to Ep 4's compaction.py. We trigger on the token count of
the *middle* — the part that actually gets summarized — not the whole call's
input. Counting the total would fire on tokens compaction can't touch (the system
prompt + the preserved recent rounds), so it could trigger with nothing to
shrink. The middle is counted with tiktoken (a real token count; for non-OpenAI
providers it's an approximation, but exact enough for a go/no-go trigger).

compact() needs an LLM to write the summary, so it takes the `client` and
`model` as arguments rather than importing them — that keeps the import one-way
(`agent → compaction`, like `agent → tools`) and avoids a circular import with
agent.py. Because they're passed in, the caller is free to summarize on a cheaper
model (even a different provider) than the main loop — agent.py resolves which
from the LLM_SUMMARIZER_* env vars.

See ../../README.md for context.
"""
import os

from tiktoken import get_encoding

# tiktoken encoder for measuring the middle's size (the part we summarize).
# cl100k_base is OpenAI's tokenizer; for Claude/others it's an approximation, but
# a real token count is plenty for a go/no-go "is the middle big enough" trigger.
_TOKENIZER = get_encoding("cl100k_base")

# --- Compaction knobs. Env-overridable; defaults shown below. Both are used in
# compact(): the threshold gates on the middle's token count, KEEP sets the tail.
# Tokens in the compactable middle before we summarize it.
COMPACTION_THRESHOLD = int(os.environ.get("COMPACTION_THRESHOLD", 25_000))
# Recent assistant rounds preserved uncompacted.
KEEP_LAST_ITERATIONS = int(os.environ.get("KEEP_LAST_ITERATIONS", 2))
# Cap on one summarizer reply. Roomy on purpose: a reasoning model's hidden thinking
# comes out of this same budget and its length is unpredictable — a tight cap would
# truncate healthy calls. This cap is also what bounds the summary that can enter
# history (one 64K "summary" once did, and got re-summarized on the next fire): a
# reply that hits the cap is never installed — the guard in compact() skips the
# round instead — so the worst case is a wasted call, not a poisoned history.
SUMMARIZER_MAX_TOKENS = int(os.environ.get("SUMMARIZER_MAX_TOKENS", 50_000))

SUMMARIZER_PROMPT = (
    "You're summarizing an in-progress coding-agent transcript so the agent can "
    "keep working with less context. Produce a concise structured summary that "
    "captures: "
    "(1) the user's original task, (2) what's been investigated so far (files read, "
    "what was found), (3) what's been changed so far (files written, edits applied), "
    "(4) what's still to do, (5) any errors encountered and how they were handled. "
    "Be terse but specific — the agent will continue from this summary, so don't "
    "omit anything that would force re-investigation."
)


def _format_as_transcript(messages):
    """Render a list of message dicts as a plain-text transcript for the
    summarizer."""
    out = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "") or ""
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                tc_lines = []
                for tc in tcs:
                    fn = tc["function"]
                    tc_lines.append(f"  → {fn['name']}({fn['arguments']})")
                out.append(f"ASSISTANT: {content}\n" + "\n".join(tc_lines))
            else:
                out.append(f"ASSISTANT: {content}")
        elif role == "tool":
            # Safeguard only: cap a single pathologically large tool output so one
            # giant dump can't blow up the summarizer call. 5K chars leaves normal
            # tool results intact, so the summarizer sees ~what the agent saw — the
            # content that actually drove the compaction trigger.
            preview = (content if len(content) < 5000
                       else content[:5000] + "...[truncated]")
            out.append(f"TOOL RESULT: {preview}")
        else:
            out.append(f"{role.upper()}: {content}")
    return "\n\n".join(out)


def _count_tokens(messages):
    """Actual token count (tiktoken) of the full content of these messages — the
    middle we'd be summarizing. Counts message content + tool-call arguments."""
    parts = []
    for m in messages:
        parts.append(str(m.get("content") or ""))
        for tc in (m.get("tool_calls") or []):
            parts.append(str(tc.get("function", {}).get("arguments", "")))
    return len(_TOKENIZER.encode("\n".join(parts)))


def build_view(history, folds):
    """Build what the model sees this turn out of the append-only `history`.

    With no folds yet, the view *is* the history. After a compaction it's the
    head (system prompt + original task), the summary that compaction produced,
    and every message appended since — rebuilt fresh each turn and never written
    back over `history`.

    A fold only has to record where the preserved tail begins, because the tail
    is always the newest messages and `history` is append-only: an index into it
    means the same thing forever. The newest fold is the only one the view
    needs, since each summary subsumes the one before it.
    """
    if not folds:
        return list(history)
    latest = folds[-1]
    return history[:2] + [latest["summary"]] + history[latest["tail_start"]:]


def compact(messages, client, model):
    """Summarize the middle of `messages`, preserving system prompt, original
    task, and the last K rounds.

    Returns (summary_msg, tail_len, in, out, middle_tokens):
      summary_msg    the summary to fold in, or None when nothing was compacted
                     (middle too small, or the guard rejected the reply)
      tail_len       how many of the newest messages the fold keeps verbatim
      in / out       summarizer tokens, spent even when the reply was rejected
      middle_tokens  the compactable middle's size (the trigger metric),
                     returned every turn so the per-iter sawtooth can be plotted

    Note what is *not* in that list: a rewritten message history. compact()
    leaves `messages` untouched and hands the summary back to the caller to
    record as a fold — that's what makes this non-destructive.
    """
    asst_positions = [i for i, m in enumerate(messages)
                      if m.get("role") == "assistant"]
    if len(asst_positions) <= KEEP_LAST_ITERATIONS:
        return None, 0, 0, 0, 0
    head = messages[:2]                            # system + original user task
    tail_start = asst_positions[-KEEP_LAST_ITERATIONS]
    middle = messages[2:tail_start]
    # The fold keeps this many newest messages.
    tail_len = len(messages) - tail_start
    if not middle:
        return None, 0, 0, 0, 0

    # Fire only when the MIDDLE (what we'd summarize) is big enough to be worth a
    # summarizer call. Counting the middle — not the total input — means we never
    # fire on tokens compaction can't shrink (head + the preserved recent rounds).
    # middle_tokens is returned every turn (fired or not) for the per-iter sawtooth.
    middle_tokens = _count_tokens(middle)
    if middle_tokens <= COMPACTION_THRESHOLD:
        return None, 0, 0, 0, middle_tokens

    summarizer_msgs = [
        {"role": "system", "content": SUMMARIZER_PROMPT},
        {"role": "user", "content": (
            f"Original task:\n{head[1]['content']}\n\n"
            f"Transcript to summarize:\n{_format_as_transcript(middle)}"
        )},
    ]
    summary_resp = client.chat.completions.create(
        model=model, messages=summarizer_msgs, max_tokens=SUMMARIZER_MAX_TOKENS,
    )
    summary_text = (summary_resp.choices[0].message.content or "").strip()
    su = summary_resp.usage
    # Guard: never fold in a bad summary. An empty reply, or one cut off by the
    # token cap (finish_reason "length" — e.g. a reasoning spiral ate the budget
    # and left a stub), is a failed attempt: record no fold this round, so the
    # next view is still the full history, and simply try again next turn (a
    # retry normally succeeds).
    if not summary_text or summary_resp.choices[0].finish_reason == "length":
        return None, 0, su.prompt_tokens, su.completion_tokens, middle_tokens
    # The summary is an ordinary message, and it takes the place of the middle
    # *in the view* — same position, so its scope is legible from where it sits.
    # It goes in as "user" because nobody actually said it: an assistant role
    # would claim the model produced it. The fold record the caller writes is
    # where the truth about its origin lives.
    summary_msg = {
        "role": "user",
        "content": (
            "[CONTEXT COMPACTED — earlier transcript summarized below.]\n\n"
            f"{summary_text}\n\n"
            "[End of summary. Continue with the most recent turns.]"
        ),
    }
    return (summary_msg, tail_len, su.prompt_tokens, su.completion_tokens,
            middle_tokens)
