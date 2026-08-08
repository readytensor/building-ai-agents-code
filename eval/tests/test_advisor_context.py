"""What the advisor is shown, now that compaction is non-destructive and both
views exist. ADVISOR_CONTEXT="view" (the default) hands it exactly what the
executor sees, folds applied; "full" hands it the whole append-only history,
trimmed to the newest ADVISOR_CONTEXT_MAX_TOKENS when it outgrows one call."""


def _state(monkeypatch, history, folds):
    import eval.agent as agent
    monkeypatch.setitem(agent._ADVISOR_STATE, "history", history)
    monkeypatch.setitem(agent._ADVISOR_STATE, "folds", folds)
    return agent


def _history(n_rounds):
    """system + task + n assistant/tool pairs."""
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "THE TASK"}]
    for i in range(n_rounds):
        msgs.append({"role": "assistant", "content": f"step {i} " + "x" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": f"result {i} " + "y" * 400})
    return msgs


SUMMARY = {"role": "user", "content": "[CONTEXT COMPACTED] ...summary..."}


def test_view_mode_applies_the_fold(monkeypatch):
    history = _history(6)
    fold = {"kind": "compaction", "tail_start": 10, "summary": SUMMARY}
    agent = _state(monkeypatch, history, [fold])
    monkeypatch.setattr(agent, "ADVISOR_CONTEXT", "view")

    shown = agent._advisor_messages()
    assert shown == history[:2] + [SUMMARY] + history[10:]
    assert len(shown) < len(history)          # the compacted middle is gone
    assert history[5] not in shown            # ...specifically


def test_full_mode_ignores_the_fold(monkeypatch):
    history = _history(6)
    fold = {"kind": "compaction", "tail_start": 10, "summary": SUMMARY}
    agent = _state(monkeypatch, history, [fold])
    monkeypatch.setattr(agent, "ADVISOR_CONTEXT", "full")
    monkeypatch.setattr(agent, "ADVISOR_CONTEXT_MAX_TOKENS", 1_000_000)

    shown = agent._advisor_messages()
    assert shown == history                   # everything, including the middle
    assert SUMMARY not in shown               # and not the summary


def test_full_mode_trims_to_the_newest_messages(monkeypatch):
    history = _history(40)
    agent = _state(monkeypatch, history, [])
    monkeypatch.setattr(agent, "ADVISOR_CONTEXT", "full")
    monkeypatch.setattr(agent, "ADVISOR_CONTEXT_MAX_TOKENS", 2_000)

    shown = agent._advisor_messages()
    assert len(shown) < len(history)                   # something was dropped
    assert shown[:2] == history[:2]                    # head always survives
    assert shown[-1] is history[-1]                    # newest always survives
    assert shown[2:] == history[len(history) - len(shown) + 2:]  # a contiguous tail


def test_view_mode_is_the_default(monkeypatch):
    """The shipped default is the bounded one; "full" is opt-in."""
    import eval.agent as agent
    assert agent.ADVISOR_CONTEXT == "view"

    history = _history(4)
    _state(monkeypatch, history, [])
    assert agent._advisor_messages() == history   # no folds yet: view == history
