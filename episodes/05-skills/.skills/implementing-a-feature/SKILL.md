---
name: implementing-a-feature
description: Use when the task adds a capability to an existing codebase. How to find the conventions the code already follows, and build the feature to match them.
---

# Implementing a feature

A codebase already has a way of doing most things. A feature that follows it is
easier to review, and more likely to be right, than one that brings a new way.

1. **Orient before you write.** Read the README, see how the project is laid out,
   and find out how its tests run. Run the suite once, so you know the starting
   state.
2. **Find the nearest example** of the kind of thing you are adding, and read it
   in full. Where is it registered? How is it named? How is it tested? Mirror all
   three.
3. **Build the smallest version** that does what the task asks, in the place its
   siblings are. Don't restructure what is already there to make room.
4. **Add a test next to the existing ones**, in their style, covering the cases the
   task names. If the task gives an example of the expected output, make that
   example a test.
5. **Run the whole suite**, not only your test. A feature that breaks a neighbor
   is not done.

Avoid: writing before reading; a general solution when the task asked for one
feature; a second way of doing something the codebase already does one way.
