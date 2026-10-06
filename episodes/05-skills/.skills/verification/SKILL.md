---
name: verification
description: Use before you say the work is complete. How to check the finished change against every requirement in the task, with evidence for each.
---

# Verification

Tests passing is one requirement among several. Before you stop, check the work
against all of them.

1. **Re-read the task** and list every requirement in it, explicit and implied.
   This list is what you verify against.
2. **For each requirement, name the evidence**: a test that exercises it and its
   output in this session, or the command you ran and what it printed. Your own
   belief that something works is not evidence.
3. **Run the project's own checks**, whole: the full test suite, not only the
   tests you wrote, and its linter if it configures one.
4. **Check the scope.** List the files you changed and make sure each change is
   one the task asked for. Remove any scratch files you created.
5. **If anything is unmet, fix it and start again from step 1.** When every
   requirement has its evidence, stop, and write a summary that lists the
   requirements with the evidence for each.
