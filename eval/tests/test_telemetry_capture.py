"""Postmortem telemetry: every tool-call record carries an excerpt of what the
tool RETURNED (a miss is often explained by the output the model saw), on both
the host and container dispatch paths. Long outputs keep head AND tail: the
head holds the session header / first error, the tail holds the verdict line
(pytest totals, the traceback's exception, the appended exit code)."""


def test_host_tool_records_full_short_output(monkeypatch):
    import eval.agent as agent
    import tools  # Ep 5's module, on sys.path via eval.agent
    from eval import container
    monkeypatch.setattr(container, "ACTIVE", None)
    monkeypatch.setattr(tools, "TOOL_CALLS", [])

    @tools.tool("echo for the test")
    def echo(text: str) -> str:
        return "short output"

    out = agent._call_tool({"echo": echo}, "echo", {"text": "hi"})
    assert out == "short output"
    record = tools.TOOL_CALLS[-1]
    assert record["tool"] == "echo"
    assert record["result_excerpt"] == "short output"  # short: kept verbatim


def test_host_tool_excerpts_long_output_head_and_tail(monkeypatch):
    import eval.agent as agent
    import tools
    from eval import container
    monkeypatch.setattr(container, "ACTIVE", None)
    monkeypatch.setattr(tools, "TOOL_CALLS", [])

    @tools.tool("echo for the test")
    def echo(text: str) -> str:
        return "H" * 350 + "T" * 350  # 700 chars: over the 600 cap

    out = agent._call_tool({"echo": echo}, "echo", {"text": "hi"})
    assert out == "H" * 350 + "T" * 350
    excerpt = tools.TOOL_CALLS[-1]["result_excerpt"]
    assert excerpt == "H" * 300 + "\n...[snip]...\n" + "T" * 300


def test_container_tool_records_result_excerpt(monkeypatch):
    import eval.agent as agent
    import tools
    from eval import container
    monkeypatch.setattr(container, "ACTIVE", "cid1")
    monkeypatch.setattr(container, "fileop",
                        lambda cid, op, kwargs: "container output")
    monkeypatch.setattr(tools, "TOOL_CALLS", [])

    out = agent._call_tool({}, "read", {"path": "f.py"})
    assert out == "container output"
    record = tools.TOOL_CALLS[-1]
    assert record["tool"] == "read"
    assert record["result_excerpt"] == "container output"
