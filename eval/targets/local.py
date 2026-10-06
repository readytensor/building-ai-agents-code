"""The `local` provider: the series' own md2html episode tasks as instances.

Each instance is one episode's task (its wording verbatim), run from that
episode's pristine initial/ tree and scored by running pytest directly:
fail_to_pass = the fixture tests the task exists to make pass (pinned by
hand from each tree's baseline), pass_to_pass = every other test in the
tree (collected automatically at load time, before any agent runs).

Episode 3's instance also carries HELD-OUT tests -- written into the working
copy at scoring time, after the agent's diff is captured. The visible toc
fixture pins the expected output by example; the held-out tests probe the
RULE behind the examples (slug normalization, dedup at depth, label-not-in-
list, href/id consistency, no-marker behavior). This mirrors SWE-bench's
held-out test_patch: the agent never sees the tests it is graded on, so it
can't tune -- or rewrite -- them.
"""
import subprocess
from pathlib import Path

from eval.scoring import score_pytest
from eval.targets import Instance

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EP3_TASK = """Our markdown documents are getting long and hard to navigate. Add a
table-of-contents extension to md2html:

- Every rendered heading gets an HTML id derived from its text; duplicate
  heading texts get distinct ids (-1, -2, ...).
- A line consisting of exactly [TOC] becomes a labeled, nested list of links
  to every heading. Without the marker: anchors only, no TOC.
- Implement it as a new extension under md2html/extensions/, registered like
  the existing ones.

I've added a fixture pair at tests/fixtures/toc.md and tests/fixtures/toc.html
showing the expected output; it currently fails. Make it pass, add your own
tests, and make sure the existing tests still pass too."""

_EP4_TASK = """I want to add support for reference-style links to our markdown
library. They look like this:

    Here is a [link][myref] in text.

    [myref]: https://example.com "Optional title"

The link definitions (the `[id]: url "title"` lines) get collected from
the document, and inline `[text][id]` references resolve to <a> elements
using those URLs. The definition lines themselves should NOT appear in
the rendered output.

This touches a few parts of the pipeline, so plan the work first and
track your progress against it as you go.

I've added a test fixture at tests/fixtures/reference_style_links.md and
tests/fixtures/reference_style_links.html showing the expected behavior;
it currently fails. Make it pass, and make sure the existing tests still
pass too."""

_EP5_TASK = """I want to add support for GitHub-flavored alerts to md2html.
They look like this:

    > [!NOTE]
    > Useful information that users should know.

    > [!WARNING]
    > Urgent info that needs immediate attention.

**IMPORTANT — read carefully:**
The test fixture at tests/fixtures/github_alerts.html may be WRONG —
I wrote it from memory and I'm not confident about the exact class
names. GitHub's actual docs are the ground truth, NOT the fixture.

You MUST use web_search to look up GitHub's latest docs FIRST to
confirm the exact class names and HTML structure. If your
implementation matches the fixture but doesn't match what GitHub
actually emits, the work is incorrect even if pytest passes. If the
docs contradict the fixture, fix the fixture to match the docs.

THEN implement the extension as a new file under md2html/extensions/.
Keep your diff minimal — don't refactor unrelated parts of the
codebase. All existing tests must still pass."""

_EP6_TASK = """I want to round out our GFM support with three more features:

  1. Strikethrough: ~~text~~ -> <del>text</del>
  2. Task lists: list items starting with `- [ ]` or `- [x]` render
     with a disabled <input type="checkbox"> prepended (checked for [x]).
  3. Autolinks: <https://example.com> -> <a href="https://example.com">\
https://example.com</a>

Add each as a new extension under md2html/extensions/ and register
each in md2html/extensions/__init__.py. There are test fixture pairs
at tests/fixtures/strikethrough.{md,html}, task_lists.{md,html}, and
autolinks.{md,html} — all three currently fail because the features
aren't implemented.

Make sure all existing tests still pass. Keep diffs minimal."""

DEFAULT_SPECS = [
    {
        # The visible toc fixture plus the held-out rule-probing tests
        # (eval/held_out/md2html__ep3-toc/, injected at scoring time).
        "id": "md2html__ep3-toc",
        "base": _REPO_ROOT / "episodes" / "03-compaction" / "initial",
        "problem_statement": _EP3_TASK,
        "fail_to_pass": [
            "tests/test_renderer.py::test_fixture_pair[toc]",
            "tests/test_toc_heldout.py::test_anchor_on_every_heading_level",
            ("tests/test_toc_heldout.py::"
             "test_slug_is_lowercase_with_punctuation_stripped"),
            ("tests/test_toc_heldout.py::"
             "test_duplicate_headings_get_distinct_sequential_ids"),
            "tests/test_toc_heldout.py::test_marker_replaced_with_labeled_nav",
            "tests/test_toc_heldout.py::test_toc_nesting_follows_heading_levels",
            "tests/test_toc_heldout.py::test_label_is_not_a_list_entry",
            "tests/test_toc_heldout.py::test_toc_hrefs_match_heading_ids_exactly",
            "tests/test_toc_heldout.py::test_no_marker_means_anchors_but_no_nav",
        ],
    },
    {
        "id": "md2html__ep4-reference-links",
        "base": _REPO_ROOT / "episodes" / "04-working-memory" / "initial",
        "problem_statement": _EP4_TASK,
        "fail_to_pass": [
            "tests/test_renderer.py::test_fixture_pair[reference_style_links]"],
    },
    {
        "id": "md2html__ep5-github-alerts",
        "base": _REPO_ROOT / "episodes" / "05-skills" / "initial",
        "problem_statement": _EP5_TASK,
        "fail_to_pass": ["tests/test_renderer.py::test_fixture_pair[github_alerts]"],
    },
    {
        "id": "md2html__ep6-gfm-trio",
        "base": _REPO_ROOT / "episodes" / "06-subagents" / "initial",
        "problem_statement": _EP6_TASK,
        "fail_to_pass": [
            "tests/test_renderer.py::test_fixture_pair[strikethrough]",
            "tests/test_renderer.py::test_fixture_pair[task_lists]",
            "tests/test_renderer.py::test_fixture_pair[autolinks]",
        ],
    },
]


# Held-out grading files live on disk, keyed by instance id (the repo's usual
# configuration-as-files idiom, like .skills/<name>/): each file under
# eval/held_out/<instance-id>/ is injected at its repo-relative path.
_HELD_OUT_DIR = Path(__file__).resolve().parents[1] / "held_out"


def _load_held_out(instance_id: str) -> dict:
    """{repo-relative-path: content} for an instance's held-out files, or {}."""
    root = _HELD_OUT_DIR / instance_id
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*")) if p.is_file()}


def make_held_out_scorer(held_out_tests: dict):
    """A scorer that injects extra test files into the working copy right
    before running pytest. The runner captures the agent's diff BEFORE verify,
    so the injected files never appear in the patch; the agent never saw them,
    so they are immune to the failure mode where an agent edits the tests it
    verifies against."""
    def scorer(repo_dir: Path, fail_to_pass: list, pass_to_pass: list):
        for relpath, content in held_out_tests.items():
            path = Path(repo_dir) / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return score_pytest(repo_dir, fail_to_pass, pass_to_pass)
    return scorer


def _collect_node_ids(base_dir: Path) -> list:
    """Every pytest node id in a base tree. Collected from the pristine tree
    at load time -- before any agent runs -- so tests an agent adds later can
    never sneak into pass_to_pass."""
    proc = subprocess.run(
        ["python", "-m", "pytest", "tests", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=base_dir, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    return [line.strip() for line in proc.stdout.splitlines() if "::" in line]


def build_instances(base_dir=None, specs=None) -> list:
    """Turn specs into Instance objects. A spec names its own base tree (or
    inherits base_dir); pass_to_pass defaults to everything the base tree's
    suite contains except the fail_to_pass ids. Raises on duplicate ids."""
    specs = DEFAULT_SPECS if specs is None else specs
    seen, instances = set(), []
    for spec in specs:
        if spec["id"] in seen:
            raise ValueError(f"duplicate instance id: {spec['id']}")
        seen.add(spec["id"])
        base = Path(spec.get("base") or base_dir)
        pass_to_pass = spec.get("pass_to_pass")
        if pass_to_pass is None:
            fail = set(spec["fail_to_pass"])
            pass_to_pass = [n for n in _collect_node_ids(base) if n not in fail]
        held_out = spec.get("held_out_tests") or _load_held_out(spec["id"])
        instances.append(Instance(
            id=spec["id"],
            problem_statement=spec["problem_statement"],
            repo_dir=base,
            fail_to_pass=spec["fail_to_pass"],
            pass_to_pass=pass_to_pass,
            scorer=make_held_out_scorer(held_out) if held_out else score_pytest,
        ))
    return instances


def get_instances() -> list:
    """Provider entry point used by the CLI."""
    return build_instances()
