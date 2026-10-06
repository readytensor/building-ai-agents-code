"""Held-out tests for the typography task.

These are the grader's tests, not the agent's: they live outside initial/ (so
they are never copied into the sandbox) and grade.py injects them into the
sandbox AFTER a run finishes. Every test checks one rule of the team's style
guide (the house-typography skill); a run that followed common defaults instead
of the guide fails the dash rules.
"""

from md2html import render


# --- Quotes -------------------------------------------------------------------

def test_double_quotes_curl():
    assert "“hello”" in render('She said "hello" to him.\n')


def test_single_quotes_and_apostrophes():
    html = render("He called it 'done' but it isn't.\n")
    assert "‘done’" in html
    assert "isn’t" in html


# --- Dashes --------------------------------------------------------------------

def test_spaced_hyphen_becomes_spaced_en_dash():
    assert "Paris – Berlin" in render("Paris - Berlin\n")


def test_double_hyphen_becomes_en_dash():
    assert "1990–1999" in render("1990--1999\n")


def test_triple_hyphen_becomes_em_dash():
    assert "wait—no" in render("wait---no\n")


def test_hyphen_inside_word_stays():
    assert "well-known" in render("a well-known rule\n")


# --- Ellipsis ------------------------------------------------------------------

def test_three_dots_become_ellipsis():
    assert "Well…" in render("Well...\n")


# --- Where the rules do not apply --------------------------------------------

def test_code_span_untouched():
    assert "<code>a -- b...</code>" in render("`a -- b...`\n")


def test_code_block_untouched():
    html = render("```\nx = \"a\" -- b...\n```\n")
    assert "--" in html
    assert "..." in html
    assert "—" not in html and "–" not in html and "…" not in html


def test_link_address_untouched():
    html = render("[x](http://a.b/c--d)\n")
    assert 'href="http://a.b/c--d"' in html


# --- Registered like the existing extensions --------------------------------

def test_extension_can_be_turned_off():
    html = render("Paris - Berlin, 1990--1999\n", extensions="")
    assert "Paris - Berlin, 1990--1999" in html
