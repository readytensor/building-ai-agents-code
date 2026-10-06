"""Tests for the repo_map orientation tool: the mapper in container_fileops
(one implementation, used in-container and host-side) and its wiring in
eval/agent.py (the repo_map tool proxy).
All offline: synthetic trees, no docker, no LLM.
"""
from pathlib import Path

from eval import container_fileops as cf


def _make_repo(tmp_path: Path) -> Path:
    """A tiny but structurally realistic project: README + runner config,
    a package with a docstring and a subpackage, a tests/ dir with a
    conftest, and junk dirs that must be skipped."""
    (tmp_path / "README.md").write_text("# Proj\n", encoding="utf-8")
    (tmp_path / "tox.ini").write_text("[tox]\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text("", encoding="utf-8")
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        '"""Top package docstring.\n\nSecond paragraph."""\n', encoding="utf-8")
    (pkg / "mod.py").write_text(
        '"""Mod doc."""\n\n'
        "class Foo(Base):\n"
        "    def bar(self, x):\n"
        "        pass\n\n"
        "def top(a, b=1, *args, **kwargs):\n"
        "    pass\n", encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "util.py").write_text("def helper():\n    pass\n",
                                         encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text("", encoding="utf-8")
    (tests / "test_mod.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    junk = tmp_path / "build"
    junk.mkdir()
    (junk / "junk.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_overview_names_readme_and_runner_config(tmp_path):
    out = cf.repo_map(_make_repo(tmp_path))
    assert "README: README.md" in out
    assert "tox.ini" in out
    assert "setup.py" in out


def test_overview_package_tree_with_docstring_one_liner(tmp_path):
    out = cf.repo_map(_make_repo(tmp_path))
    assert "pkg/ (2 modules) -- Top package docstring." in out
    assert "pkg/sub/ (2 modules)" in out


def test_overview_lists_test_locations(tmp_path):
    out = cf.repo_map(_make_repo(tmp_path))
    assert "tests/" in out.split("Test locations")[1]


def test_overview_skips_junk_dirs(tmp_path):
    out = cf.repo_map(_make_repo(tmp_path))
    assert "build/" not in out


def test_subtree_shows_signatures(tmp_path):
    out = cf.repo_map(_make_repo(tmp_path), "pkg")
    assert "pkg/mod.py -- Mod doc." in out
    assert "class Foo(Base)" in out
    assert "def bar(self, x)" in out
    assert "def top(a, b, *args, **kwargs)" in out
    assert "pkg/sub/util.py" in out  # recursive


def test_subtree_tolerates_syntax_errors(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "pkg" / "broken.py").write_text("def (\n", encoding="utf-8")
    out = cf.repo_map(repo, "pkg")
    assert "pkg/broken.py (could not parse)" in out
    assert "class Foo(Base)" in out  # the rest still mapped


def test_repo_map_on_a_file_is_an_error(tmp_path):
    repo = _make_repo(tmp_path)
    out = cf.repo_map(repo, "pkg/mod.py")
    assert out.startswith("Error:")


def test_repo_map_dispatches_as_an_op(tmp_path):
    out = cf.dispatch({"op": "repo_map", "path": "."}, root=_make_repo(tmp_path))
    assert "README: README.md" in out


def test_overview_truncates_huge_package_trees(tmp_path):
    repo = tmp_path
    for i in range(cf._DIR_CAP + 10):
        d = repo / f"pkg{i:03d}"
        d.mkdir()
        (d / "__init__.py").write_text("", encoding="utf-8")
    out = cf.repo_map(repo)
    assert "more directories" in out


def test_top_level_files_are_capped(tmp_path):
    for i in range(cf._TOPFILE_CAP + 20):
        (tmp_path / f"file{i:02d}.txt").write_text("", encoding="utf-8")
    out = cf.repo_map(tmp_path)
    assert "(+20 more)" in out


def test_output_is_hard_capped(tmp_path):
    # 300 long-named packages with long docstrings: line caps alone would
    # still emit ~20K chars; the hard cap must kick in.
    for i in range(300):
        d = tmp_path / f"averylongpackagedirectoryname{i:04d}"
        d.mkdir()
        (d / "__init__.py").write_text('"""' + "x" * 90 + '"""\n', encoding="utf-8")
    out = cf.repo_map(tmp_path)
    assert len(out) <= cf._MAP_CHAR_CAP + 100
    assert "map truncated" in out


def test_subtree_caps_files_and_signature_lines(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for i in range(cf._FILE_CAP + 5):
        (pkg / f"m{i:03d}.py").write_text("def f():\n    pass\n", encoding="utf-8")
    out = cf.repo_map(tmp_path, "pkg")
    assert "5 more modules not shown" in out


def test_subtree_truncates_at_sig_cap(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    body = "".join(f"def f{i:04d}():\n    pass\n" for i in range(cf._SIG_CAP + 10))
    (pkg / "big.py").write_text(body, encoding="utf-8")
    out = cf.repo_map(tmp_path, "pkg")
    assert f"truncated at {cf._SIG_CAP} lines" in out


def test_testbed_paths_are_accepted(tmp_path):
    # The container-facing contract: models copy /testbed/... paths from
    # shell output; _safe_path translates them.
    repo = _make_repo(tmp_path)
    out = cf.repo_map(repo, "/testbed/pkg")
    assert "class Foo(Base)" in out


def test_a_directory_named_readme_is_not_readme_evidence(tmp_path):
    (tmp_path / "README").mkdir()
    (tmp_path / "somefile.txt").write_text("", encoding="utf-8")
    out = cf.repo_map(tmp_path)
    assert "README: (none found)" in out


def test_async_functions_are_labeled(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "aio.py").write_text("async def go(x):\n    pass\n", encoding="utf-8")
    out = cf.repo_map(tmp_path, "pkg")
    assert "async def go(x)" in out


# --- agent wiring (eval/agent.py) -----------------------------------------
# Tests cover the drill-down tool proxy only (the injected root map was
# removed -- orientation is something the agent does by calling tools).
# Importing eval.agent is offline-safe (test_bash_proxy.py precedent): it
# builds an OpenAI client only inside solve().

def test_repo_map_tool_routes_to_container(monkeypatch):
    from eval import agent, container
    seen = {}

    def fake_fileop(cid, op, kwargs):
        seen["call"] = (cid, op, kwargs)
        return "MAPPED"

    monkeypatch.setattr(container, "ACTIVE", "cid123")
    monkeypatch.setattr(container, "fileop", fake_fileop)
    assert agent.repo_map.__wrapped__("pkg") == "MAPPED"
    assert seen["call"] == ("cid123", "repo_map", {"path": "pkg"})


def test_repo_map_tool_host_path(tmp_path, monkeypatch):
    from eval import agent, container
    import tools  # episodes/05-skills, put on sys.path by eval.agent
    monkeypatch.setattr(container, "ACTIVE", None)
    monkeypatch.setattr(tools, "SANDBOX", tmp_path)
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    out = agent.repo_map.__wrapped__(".")
    assert "README: README.md" in out


def test_repo_map_is_in_the_toolset():
    from eval import agent
    assert agent.repo_map.tool_definition["function"]["name"] == "repo_map"


def test_huge_tree_collapses_to_depth_2_and_keeps_all_top_branches(tmp_path):
    # 200 leaf dirs under aa/, but only a handful of top branches. Flat
    # rendering would burn the cap on aa/* and truncate zz/ out; the
    # collapsed rendering must show every top branch, aggregated.
    for i in range(cf._DIR_CAP + 50):
        d = tmp_path / "aa" / f"leaf{i:03d}"
        d.mkdir(parents=True)
        (d / "mod.py").write_text("x = 1\n", encoding="utf-8")
    zz = tmp_path / "zz"
    zz.mkdir()
    (zz / "core.py").write_text("x = 1\n", encoding="utf-8")
    out = cf.repo_map(tmp_path)
    assert f"aa/ ({cf._DIR_CAP + 50} modules in {cf._DIR_CAP + 50} dirs)" in out
    assert "zz/ (1 modules)" in out
    assert "leaf000" not in out  # depth-3 paths are collapsed away


def test_small_tree_stays_fully_detailed(tmp_path):
    out = cf.repo_map(_make_repo(tmp_path))
    assert "pkg/sub/ (2 modules)" in out  # no collapse below the cap


def test_collapse_falls_back_to_depth_1(tmp_path):
    # More than _DIR_CAP distinct depth-2 prefixes: depth-2 rendering is
    # still over the cap, so it must fall back to depth 1.
    for i in range(cf._DIR_CAP + 10):
        d = tmp_path / "pkg" / f"sub{i:03d}" / "inner"
        d.mkdir(parents=True)
        (d / "mod.py").write_text("x = 1\n", encoding="utf-8")
    out = cf.repo_map(tmp_path)
    assert f"pkg/ ({cf._DIR_CAP + 10} modules in {cf._DIR_CAP + 10} dirs)" in out
    assert "sub000" not in out
