"""container_fileops runs inside the instance container; offline it is a plain
module, exercised here against a tmp root. Semantics must mirror the episodes'
host file tools exactly -- the model sees the same tools on both paths."""
import pytest

from eval.container_fileops import dispatch


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("alpha = 1\nbeta = 2\ngamma = 3\n",
                                             encoding="utf-8")
    return tmp_path


def test_read_numbers_lines(root):
    out = dispatch({"op": "read", "path": "pkg/mod.py"}, root=root)
    assert "    1\talpha = 1" in out
    assert "    3\tgamma = 3" in out


def test_read_slice_keeps_real_line_numbers(root):
    out = dispatch({"op": "read", "path": "pkg/mod.py", "offset": 2, "limit": 1},
                   root=root)
    assert "    2\tbeta = 2" in out
    assert "alpha" not in out
    assert "(showing lines 2-2 of 3)" in out


def test_read_missing_file(root):
    assert (dispatch({"op": "read", "path": "nope.py"}, root=root)
            == "Error: nope.py does not exist.")


def test_testbed_absolute_paths_are_accepted(root):
    out = dispatch({"op": "read", "path": "/testbed/pkg/mod.py"}, root=root)
    assert "alpha = 1" in out
    listing = dispatch({"op": "list_files", "path": "/testbed"}, root=root)
    assert "mod.py" in listing


def test_escaping_paths_come_back_as_error_strings(root):
    out = dispatch({"op": "read", "path": "../outside.py"}, root=root)
    assert out.startswith("Error executing read")


def test_write_creates_parents(root):
    out = dispatch({"op": "write", "path": "new/dir/f.txt", "content": "hi"},
                   root=root)
    assert out == "Wrote 2 bytes to new/dir/f.txt."
    assert (root / "new" / "dir" / "f.txt").read_text() == "hi"


def test_edit_requires_unique_match(root):
    (root / "e.txt").write_text("x\nx\n", encoding="utf-8")
    out = dispatch({"op": "edit", "path": "e.txt", "old_string": "x",
                    "new_string": "y"}, root=root)
    assert "appears 2 times" in out
    out = dispatch({"op": "edit", "path": "e.txt", "old_string": "x",
                    "new_string": "y", "replace_all": True}, root=root)
    assert out == "Replaced 2 occurrence(s) in e.txt."
    assert (root / "e.txt").read_text() == "y\ny\n"


def test_grep_skips_noise_dirs(root):
    (root / "build").mkdir()
    (root / "build" / "junk.py").write_text("alpha = 9\n", encoding="utf-8")
    out = dispatch({"op": "grep", "pattern": "alpha"}, root=root)
    assert "mod.py" in out
    assert "junk" not in out


def test_unknown_op_is_an_error_string(root):
    out = dispatch({"op": "explode"}, root=root)
    assert out.startswith("Error executing explode")


def test_bad_kwarg_is_an_error_string(root):
    out = dispatch({"op": "read", "path": "pkg/mod.py", "bogus": 1}, root=root)
    assert out.startswith("Error executing read: TypeError")
