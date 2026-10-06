"""Held-out tests for the alerts task.

These are the grader's tests, not the agent's: they live outside initial/ (so
they are never copied into the sandbox) and grade.py injects them into the
sandbox AFTER a run finishes. Every test checks a rule the task states; the
visible fixture shows one worked example of each rule, these probe the rule.
"""

from md2html import render

TYPES = ["NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION"]


def alert(kind: str, *lines: str) -> str:
    return "\n".join([f"> [!{kind}]", *(f"> {line}" for line in lines)]) + "\n"


# --- The five types ----------------------------------------------------------

def test_each_type_gets_its_class_and_title():
    for kind in TYPES:
        html = render(alert(kind, "Body text."))
        assert f'class="markdown-alert markdown-alert-{kind.lower()}"' in html
        assert (f'<p class="markdown-alert-title">{kind.capitalize()}</p>'
                in html)
        assert "<blockquote>" not in html


def test_alert_body_renders_like_blockquote_content():
    html = render(alert("TIP", "Use **bold** and `code`."))
    assert "<strong>bold</strong>" in html
    assert "<code>code</code>" in html


def test_alert_can_hold_several_paragraphs():
    html = render(alert("NOTE", "First paragraph.", "", "Second paragraph."))
    body = html.split("markdown-alert-title")[1]
    assert body.count("<p>") == 2


# --- What stays an ordinary blockquote --------------------------------------

def test_unknown_type_stays_blockquote():
    html = render(alert("DANGER", "Not a GitHub alert type."))
    assert "<blockquote>" in html
    assert "markdown-alert" not in html


def test_marker_followed_by_text_stays_blockquote():
    html = render("> [!NOTE] Heads up.\n")
    assert "<blockquote>" in html
    assert "markdown-alert" not in html


def test_plain_blockquote_unchanged():
    html = render("> Just a quote.\n")
    assert "<blockquote>" in html
    assert "<p>Just a quote.</p>" in html
    assert "markdown-alert" not in html


# --- Registered like the existing extensions --------------------------------

def test_extension_can_be_turned_off():
    html = render(alert("WARNING", "Off."), extensions="")
    assert "<blockquote>" in html
    assert "markdown-alert" not in html
