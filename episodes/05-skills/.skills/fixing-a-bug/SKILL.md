---
name: fixing-a-bug
description: Use when the task reports behavior that is wrong. How to reproduce a bug, find its cause, and fix it without breaking what works.
---

# Fixing a bug

1. **Reproduce it first.** Write a failing test that shows the bug, in the style
   of the project's suite. If you can't make it fail, you don't understand the
   bug yet.
2. **Locate the cause.** Follow the data from where the symptom shows to where
   the wrong value is made: search for the text of the symptom, then read the
   code path that produces it.
3. **Fix the cause, not the symptom.** Make the smallest change that corrects the
   behavior; don't patch around it at the place it was noticed.
4. **Confirm both ways.** Your reproduction now passes, and the whole suite is
   still green.
5. **Look for the same mistake elsewhere.** Search for the pattern you fixed; a
   bug often has siblings.
