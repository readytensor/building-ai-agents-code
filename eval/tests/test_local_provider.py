import pytest

from eval.targets.local import build_instances


def test_build_instances_produces_wellformed_instances(tmp_path):
    # A stand-in md2html base: enough that Instance fields are exercised.
    base = tmp_path / "md2html_base"
    base.mkdir()
    (base / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (base / "test_math.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )

    specs = [{
        "id": "md2html__demo",
        "problem_statement": "Fix add.",
        "fail_to_pass": ["test_math.py::test_add"],
        "pass_to_pass": [],
    }]
    instances = build_instances(base_dir=base, specs=specs)

    assert len(instances) == 1
    inst = instances[0]
    assert inst.id == "md2html__demo"
    assert inst.repo_dir == base
    assert inst.fail_to_pass == ["test_math.py::test_add"]
    assert callable(inst.scorer)


def test_default_specs_are_the_episode_tasks():
    # The real seeded pool: Eps 3-6 tasks over their pristine initial/ trees.
    # Collects each tree's suite (subprocess pytest --collect-only, no LLM).
    instances = build_instances()
    by_id = {i.id: i for i in instances}
    assert set(by_id) == {
        "md2html__ep3-toc", "md2html__ep4-reference-links",
        "md2html__ep5-github-alerts", "md2html__ep6-gfm-trio",
    }
    for inst in instances:
        assert inst.repo_dir.is_dir(), f"{inst.id}: base tree missing"
        # Each tree's suite has 40+ tests; a tiny count means collection broke.
        assert len(inst.pass_to_pass) >= 40, f"{inst.id}: suspiciously few P2P ids"
        assert not set(inst.fail_to_pass) & set(inst.pass_to_pass)
    # The failing fixture tests are pinned per episode; ep3 additionally grades
    # against held-out rule-probing tests (injected at scoring time), so its
    # fail_to_pass mixes the visible fixture with the held-out node ids.
    ep3_f2p = by_id["md2html__ep3-toc"].fail_to_pass
    assert ep3_f2p[0] == "tests/test_renderer.py::test_fixture_pair[toc]"
    assert all(n.startswith("tests/test_toc_heldout.py::") for n in ep3_f2p[1:])
    assert len(ep3_f2p) == 9
    assert len(by_id["md2html__ep6-gfm-trio"].fail_to_pass) == 3


def test_held_out_files_are_discovered_by_instance_id():
    # Held-out grading files live under eval/held_out/<instance-id>/, keyed by
    # repo-relative path -- the configuration-as-files convention.
    from eval.targets.local import _load_held_out
    files = _load_held_out("md2html__ep3-toc")
    assert list(files) == ["tests/test_toc_heldout.py"]
    assert "def test_" in files["tests/test_toc_heldout.py"]
    assert _load_held_out("no-such-instance") == {}


def test_held_out_tests_are_injected_at_scoring_time(tmp_path):
    # Working copy as the agent left it: renamed correctly, no test for it.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.py").write_text("class ASTNode:\n    pass\n")

    held_out = {"test_held_out.py": (
        "def test_renamed():\n"
        "    from lib import ASTNode  # noqa: F401\n"
        "    import lib\n"
        "    assert not hasattr(lib, 'Node')\n"
    )}
    from eval.targets.local import make_held_out_scorer
    scorer = make_held_out_scorer(held_out)
    verdict = scorer(repo, ["test_held_out.py::test_renamed"], [])
    assert verdict.passed is True
    assert (repo / "test_held_out.py").exists()  # injected, not pre-existing


def test_held_out_tests_fail_on_unchanged_base(tmp_path):
    # A no-op agent must NOT pass: the held-out test fails on the base state.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.py").write_text("class Node:\n    pass\n")

    held_out = {"test_held_out.py": (
        "def test_renamed():\n"
        "    from lib import ASTNode  # noqa: F401\n"
    )}
    from eval.targets.local import make_held_out_scorer
    scorer = make_held_out_scorer(held_out)
    verdict = scorer(repo, ["test_held_out.py::test_renamed"], [])
    assert verdict.passed is False


def test_ids_must_be_unique():
    specs = [
        {"id": "dup", "problem_statement": "", "fail_to_pass": [],
         "pass_to_pass": []},
        {"id": "dup", "problem_statement": "", "fail_to_pass": [],
         "pass_to_pass": []},
    ]
    with pytest.raises(ValueError):
        build_instances(base_dir=".", specs=specs)
