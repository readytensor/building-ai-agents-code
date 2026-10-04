# Agents from First Principles

Build a working coding agent from scratch: from a plain `while` loop with a single tool, all the way to multiple agents working in parallel. No frameworks, no magic: just Python and a model API.

> [!NOTE]
> This page was rendered by `md2html`, the very tool you build the agent around in this series. If you're reading the HTML version, the worked example just rendered its own series description.

This is companion material for the **Agents from First Principles** video series. Each episode adds *one idea*, in code, on top of the last, and the diff between one episode and the next is the lesson.

---

## What you'll be able to do

By the end, you'll have assembled the agent's full set of capabilities, one at a time:

- [x] Run a minimal agent loop, and understand what an "agent" actually is
- [x] Give it general tools, and design good ones
- [x] Keep long tasks affordable with compaction
- [x] Give it durable working memory that survives compaction
- [x] Load capabilities on demand with a skills system
- [x] Split independent work across parallel subagents
- [ ] (bonus, deferred) Apply the whole thing to a real, non-toy project

---

## The series

Episodes are named for the **mechanism** each one builds:

| #  | Episode         | The question                          | What you build                                |
|----|-----------------|---------------------------------------|-----------------------------------------------|
| 1  | The Loop        | What *is* an agent?                   | A `while` loop + one `bash` tool              |
| 2  | Tools           | How does it *do* things?              | `read` / `write` / `edit` / `grep` + `@tool`  |
| 3  | Compaction      | Why does it get worse on long tasks?  | Rolling-summary compaction                    |
| 4  | Working Memory  | How does it stay on track?            | A durable plan that survives compaction       |
| 5  | Skills          | How does it reach beyond a fixed kit? | Lazy-loaded `SKILL.md` capabilities           |
| 6  | Subagents       | When is one agent the wrong shape?    | `delegate` + parallel worker agents           |

Each episode follows one rhythm: one question, one limitation, one addition in code, one before-and-after.

---

## How it's taught

The worked example is a **coding agent**, the cleanest domain to learn in: a tight feedback loop and a small tool surface. It works on `md2html`, a small Markdown-to-HTML library[^toy] with real module boundaries (lexer, parser, renderer, extensions).

The whole agent, in spirit, is this:

```python
while True:
    reply = model(messages, tools)
    if not reply.tool_calls:
        break                       # natural stop: nothing left to do
    for call in reply.tool_calls:
        messages.append(run(call))  # do the work, feed the result back
```

That loop *is* Episode 1. Everything after is one deliberate addition, made only when a concrete limitation forces it. ~~Heavyweight frameworks~~ not required.

> The big claim of the series: an agent is a loop around a model that can call tools. Everything else (context, memory, skills, coordination) is something you add on purpose.

---

## Who it's for

Engineers comfortable with Python and calling an LLM API. No prior agent-building experience assumed. See the repo's `README.md` for setup, and the source at <https://github.com/readytensor/building-agents-code>.

For the reasoning behind the build order, see the [first-principles approach][fp]: start from the loop, and add only what a real limitation demands.

[fp]: https://github.com/readytensor/building-agents-code "Building Agents from First Principles"

[^toy]: `md2html` is small enough to read in one sitting, but structured enough that every episode's task lands on a real seam rather than an arbitrary split.
