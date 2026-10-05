"""The system prompt is one shared artifact, kept as system_prompt.md files:
a common core in every episode, mechanism sections appearing as episodes
introduce them (Working plan in Ep 4, Skills in Ep 5), and the eval agent
byte-identical to Ep 5. This test is the drift guard: any copy edited
alone fails here.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _prompt(rel: str) -> str:
    return (_ROOT / rel / "system_prompt.md").read_text(encoding="utf-8")


def _sections(text: str) -> dict:
    """Split a prompt into {header: body}; the preamble/closing get key ''."""
    parts, key = {"": []}, ""
    for line in text.splitlines():
        if line.startswith("## "):
            key = line.strip()
            parts[key] = []
        else:
            parts[key].append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}


def test_eval_prompt_is_byte_identical_to_ep5():
    # The eval agent IS the Ep 5 agent; its prompt carries no eval-specific
    # sections. (An Orientation section existed briefly on 2026-07-04:
    # mandating repo_map bought first-call adoption but no verdict change,
    # so the tool is now merely available and the agent uses it as it
    # sees fit.)
    assert _prompt("eval") == _prompt("episodes/05-skills")


def test_early_episodes_share_the_same_core():
    assert (_prompt("episodes/03-loop")
            == _prompt("episodes/04-tools")
            == _prompt("episodes/03-compaction"))


def test_later_episodes_only_add_their_mechanism_section():
    core = _sections(_prompt("episodes/03-loop"))
    ep4 = _sections(_prompt("episodes/04-working-memory"))
    ep5 = _sections(_prompt("episodes/05-skills"))
    for header, body in core.items():
        if header:  # the '' preamble is checked below
            assert ep4[header] == body, f"ep4 drifted in {header}"
            assert ep5[header] == body, f"ep5 drifted in {header}"
    assert set(ep4) - set(core) == {"## Working plan"}
    assert set(ep5) - set(ep4) == {"## Skills"}
    assert ep5["## Working plan"] == ep4["## Working plan"]


def test_preamble_is_common():
    # The identity/task preamble (before the first ## section) must be
    # identical everywhere. The closing text rides inside the last section's
    # body, so the section comparison above already covers it.
    core = _sections(_prompt("episodes/03-loop"))[""]
    assert _sections(_prompt("episodes/04-working-memory"))[""] == core
    assert _sections(_prompt("episodes/05-skills"))[""] == core
