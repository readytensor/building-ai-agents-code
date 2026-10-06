# eval/ - reference-agent evaluation

Runs the reference coding agent over a pool of problems and reports how often it
resolves them, so we can see that it generalizes beyond any single example.

## Run it

Token-free smoke check of the harness (fake agents over a synthetic repo, no
API calls):

    python -m pytest eval/tests -q

A real run of the reference agent over SWE-bench Verified instances (needs
`pip install datasets`, plus Docker and WSL). The agent works directly inside
each instance's official container image, on its own `/testbed` checkout with
the repo's real frozen environment; each sample is graded officially right
after it runs:

    python -m eval.run_eval --source swebench --n 5

Useful flags:

- `--n N` how many instances to sample
- `--stratified` spread `--n` across the easy/medium/hard buckets in
  proportion to their share of the pool, so a small sample mirrors the
  benchmark's difficulty mix (e.g. `--n 20 --stratified` picks 8/10/2)
- `--id <instance-id>` run one specific instance
- `--difficulty {easy,medium,hard}` filter by Verified's time-to-fix bucket
- `--repo <substring>` filter by repository, e.g. `--repo django`
- `--no-grade` skip the per-sample official grading (on by default for
  swebench; grading a sample right away surfaces a broken setup at sample 1)
- `--no-clean-images` keep each instance's Docker image after its sample
  finishes (removal is on by default so a batch's disk use stays flat)
- `--repeat K` run each sampled instance K times (for pass@k and variance)
- `--seed S` reproducible sampling
- `--keep {none,failures,all}` how much per-instance log detail to retain

For bigger batches, `eval/dispatch.py` runs one `run_eval` subprocess per
instance, at most `--workers` at a time, and consolidates one summary at the
end. A killed batch resumes where it left off (completed instances are
skipped). It takes an explicit instance list:

    python -m eval.dispatch --batch my-batch --workers 3 --ids id1 id2 ...

The `local` provider (`--source local`) runs the series' own episode tasks
(Eps 3-6) over each episode's pristine `initial/` md2html tree, scored by
running that tree's pytest suite directly (no Docker needed):

    python -m eval.run_eval --source local --n 4

## Results

Each run writes a timestamped batch under `eval/results/<timestamp>/`
(summary, per-instance verdicts and diffs) and appends one row to
`eval/results/scoreboard.jsonl` (rendered as `scoreboard.md`) so runs are easy
to compare. Verbose logs are kept only for failed instances by default.

## Tests

    python -m pytest eval/tests -q

The suite runs offline in a few seconds: it exercises the whole pipeline with
fake agents over a tiny synthetic repo, so it never calls an LLM.
