import pytest

from eval.fake_agents import fixing_solver, git, noop_solver

# A minimal "buggy" project: add() is wrong, so test_add fails (FAIL_TO_PASS);
# test_mul passes and must stay passing (PASS_TO_PASS).
BASE_FILES = {
    "calc.py": ("def add(a, b):\n    return a - b\n\n\n"
                "def mul(a, b):\n    return a * b\n"),
    "test_math.py": (
        "from calc import add, mul\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_mul():\n    assert mul(2, 3) == 6\n"
    ),
}

FAIL_TO_PASS = ["test_math.py::test_add"]
PASS_TO_PASS = ["test_math.py::test_mul"]


@pytest.fixture
def base_repo(tmp_path):
    """A committed git working copy in its base (failing) state."""
    repo = tmp_path / "base"
    repo.mkdir()
    for name, content in BASE_FILES.items():
        (repo / name).write_text(content, encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


@pytest.fixture
def solvers():
    return {"fixing": fixing_solver, "noop": noop_solver}
