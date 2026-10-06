# Release notes

"This release," the team wrote, "is about the small things." It's the first
one since 1990--1999 - a long gap... and a long list.

## Changes

- Faster rendering - about twice the speed on long documents.
- Fixed a crash on empty tables---the parser now skips them.
- The command line accepts a range of files: `notes-01.md -- notes-09.md`.

## In code, nothing changes

```sh
md2html "notes.md" -o out.html --stdout   # quotes and dashes stay as typed
```

The address in a link stays as typed too: [the changelog](http://example.com/a--b).
