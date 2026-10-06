"""Pre-acceptance audit: the environment's half of the stop handshake.

The agent loop has always ended when the model returns a turn with no tool
calls -- the model REQUESTS the stop. Until now the environment rubber-stamped
every request. These checks are the environment's right to refuse: mechanical
contract checks over the captured patch text, run at the exit door of the loop
(see solve() in eval/agent.py), with one bounce-back before the stop is
accepted unconditionally.

Contract checks, never quality judgments: the audit can say "your patch is
empty" or "you rewrote an existing test"; it never says "your fix looks
wrong" -- solution quality stays the agent's job. Each check earns its place
with a real lost run:
  - empty patch: matplotlib-25960 bailed empty mid-investigation despite the
    prompt's no-empty-finish rule (prompt rules are probabilistic; gates are
    not), and matplotlib-25332's work sat invisible after an in-container
    `git commit` until diff capture was base-pinned.
  - modified existing tests: DeepSeek once edited a failing test file 6x to
    green an over-broad patch (sphinx-9602 v1); grading resets tests, so the
    "fix" graded with 2 regressions. Pure additions stay allowed -- gold
    patches add tests to existing files all the time.
  - binary artifacts: a .coverage file in django-11400's patch aborted the
    grader's `git apply`, so the tests ran against an UNPATCHED tree.

All functions are pure over unified-diff text: no docker, no filesystem.
"""


def _file_blocks(diff: str):
    """Split a unified diff into (path, is_new_file, is_deleted, lines) blocks."""
    blocks = []
    current = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            # "diff --git a/path b/path" -- take the b/ side (post-image).
            path = line.split(" b/", 1)[-1]
            current = {"path": path, "new": False, "deleted": False, "lines": []}
        elif current is not None:
            if line.startswith("new file mode"):
                current["new"] = True
            elif line.startswith("deleted file mode"):
                current["deleted"] = True
            current["lines"].append(line)
    if current:
        blocks.append(current)
    return blocks


def _is_test_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return ("tests" in parts or "test" in parts
            or name.startswith("test_") or name.endswith("_test.py"))


def check_empty(diff: str) -> list:
    if diff.strip():
        return []
    return ["the captured patch is empty: no code changes will be submitted. "
            "If your work is stashed (`git stash`) or committed, restore it to "
            "the working tree; if the task is not finished, continue working."]


def check_test_modifications(diff: str) -> list:
    """Existing test files with deleted/changed lines (or deleted outright).
    New test files and pure additions to existing ones never bounce. Edits to
    existing tests bounce once as a confirmation tripwire, not a prohibition:
    the prompt allows surgical amendments that reflect the task's own behavior
    change, and a justified amendment survives the bounce (the agent keeps it
    and finishes; the stop is then accepted)."""
    findings = []
    for block in _file_blocks(diff):
        if not _is_test_path(block["path"]) or block["new"]:
            continue
        if block["deleted"]:
            findings.append(f"existing test file deleted: {block['path']}. "
                            "Delete tests only if the behavior change your "
                            "task calls for requires it; otherwise restore "
                            "the file.")
            continue
        removed = any(line.startswith("-") and not line.startswith("---")
                      for line in block["lines"])
        if removed:
            findings.append(f"existing test lines modified in: {block['path']}. "
                            "Keep these edits only if they reflect the "
                            "behavior change your task calls for; otherwise "
                            "restore them and add new tests instead.")
    return findings


def check_binary_files(diff: str) -> list:
    findings = []
    for block in _file_blocks(diff):
        if any(line.startswith("Binary files ") or line == "GIT binary patch"
               for line in block["lines"]):
            findings.append(f"binary file in the patch: {block['path']}. "
                            "Remove it -- it is likely a test-run artifact, "
                            "and binary content cannot be reviewed or applied "
                            "as a patch.")
    return findings


def run_checks(diff: str) -> list:
    """All findings for one captured patch; empty list = stop accepted."""
    return (check_empty(diff) + check_test_modifications(diff)
            + check_binary_files(diff))
