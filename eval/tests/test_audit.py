"""The pre-acceptance audit: mechanical contract checks at the loop's exit
door, with one bounce-back. Check functions are pure over diff text; the
loop integration runs token-free against a scripted fake chat client.
"""
import json
from types import SimpleNamespace

from eval import audit

# --- diff fixtures -----------------------------------------------------------

SOURCE_EDIT = """\
diff --git a/pkg/core.py b/pkg/core.py
index 111..222 100644
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,3 +1,3 @@
-def f(): return 1
+def f(): return 2
"""

TEST_EDIT = """\
diff --git a/tests/test_core.py b/tests/test_core.py
index 111..222 100644
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,3 +1,3 @@
-    assert f() == 1
+    assert f() == 2
"""

TEST_ADDITION = """\
diff --git a/tests/test_core.py b/tests/test_core.py
index 111..222 100644
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -3,0 +4,2 @@
+def test_new_case():
+    assert f() == 2
"""

NEW_TEST_FILE = """\
diff --git a/tests/test_extra.py b/tests/test_extra.py
new file mode 100644
index 000..222
--- /dev/null
+++ b/tests/test_extra.py
@@ -0,0 +1,2 @@
+def test_extra():
+    assert True
"""

DELETED_TEST_FILE = """\
diff --git a/tests/test_core.py b/tests/test_core.py
deleted file mode 100644
index 111..000
--- a/tests/test_core.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def test_old():
-    assert f() == 1
"""

BINARY_ARTIFACT = """\
diff --git a/.coverage b/.coverage
index 111..222 100644
Binary files a/.coverage and b/.coverage differ
"""


# --- check functions ---------------------------------------------------------

def test_empty_patch_is_a_finding():
    findings = audit.run_checks("   \n")
    assert len(findings) == 1
    assert "empty" in findings[0]
    # the recovery hint names the failure modes seen live
    assert "git stash" in findings[0]


def test_source_edit_passes_clean():
    assert audit.run_checks(SOURCE_EDIT) == []


def test_modified_existing_test_lines_are_a_finding():
    # A confirmation tripwire, not a prohibition: the message asks the agent
    # to keep the edit only if the task's behavior change requires it.
    findings = audit.run_checks(TEST_EDIT)
    assert len(findings) == 1
    assert "tests/test_core.py" in findings[0]
    assert "behavior change" in findings[0]


def test_pure_additions_to_existing_tests_are_allowed():
    # Gold patches add tests to existing files all the time; the contract
    # only forbids weakening what already exists.
    assert audit.run_checks(TEST_ADDITION) == []


def test_new_test_files_are_allowed():
    assert audit.run_checks(NEW_TEST_FILE) == []


def test_deleted_test_file_is_a_finding():
    findings = audit.run_checks(DELETED_TEST_FILE)
    assert len(findings) == 1
    assert "deleted" in findings[0]


def test_binary_artifact_is_a_finding():
    findings = audit.run_checks(BINARY_ARTIFACT)
    assert len(findings) == 1
    assert ".coverage" in findings[0]


def test_findings_compose_across_checks():
    findings = audit.run_checks(TEST_EDIT + BINARY_ARTIFACT)
    assert len(findings) == 2


# --- the stop handshake in solve() -------------------------------------------

class _StopMsg:
    """An assistant turn with no tool calls: the model requesting a stop."""
    tool_calls = None
    content = "I'm done."

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


def _stop_response():
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=_StopMsg())])


def _run_solve(tmp_path, monkeypatch, audit_hook):
    """Drive eval.agent.solve token-free: every model turn requests a stop."""
    from eval import agent

    chat_calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent, "_chat_with_retry",
                        lambda client, **kw:
                        chat_calls.append(kw) or _stop_response())
    monkeypatch.chdir(tmp_path)  # solve writes telemetry to cwd
    agent.solve(tmp_path, "task", audit=audit_hook)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    transcript = json.loads(
        (tmp_path / "transcript.json").read_text(encoding="utf-8"))
    return chat_calls, metrics["agents"][0]["audit"], transcript


def test_clean_stop_is_accepted_first_time(tmp_path, monkeypatch):
    audit_calls = []
    hook = lambda: audit_calls.append(1) or []  # noqa: E731
    chat_calls, audit_metrics, _ = _run_solve(tmp_path, monkeypatch, hook)
    assert len(chat_calls) == 1          # one turn: stop requested, granted
    assert len(audit_calls) == 1
    assert audit_metrics == {"bounces": 0, "unresolved": []}


def test_findings_bounce_once_then_stop_is_unconditional(tmp_path, monkeypatch):
    audit_calls = []
    hook = lambda: (  # noqa: E731
        audit_calls.append(1) or ["the captured patch is empty"])
    chat_calls, audit_metrics, transcript = _run_solve(tmp_path, monkeypatch, hook)
    # Stop requested twice: first bounced with the finding, second accepted
    # unconditionally even though the finding still stands.
    assert len(chat_calls) == 2
    assert len(audit_calls) == 2
    assert audit_metrics["bounces"] == 1
    assert audit_metrics["unresolved"] == ["the captured patch is empty"]
    bounce = [m for m in transcript if m.get("role") == "user"
              and str(m.get("content", "")).startswith("AUDIT:")]
    assert len(bounce) == 1
    assert "empty" in bounce[0]["content"]


def test_no_hook_keeps_the_classic_natural_stop(tmp_path, monkeypatch):
    chat_calls, audit_metrics, _ = _run_solve(tmp_path, monkeypatch, None)
    assert len(chat_calls) == 1
    assert audit_metrics == {"bounces": 0, "unresolved": []}
