# scripts/

Helper scripts for TACC job submission. Deployment entrypoints currently live in
`deploy/tacc/`; move or wrap them here only if that simplifies team usage.

`python -m scripts.replay_agent_finalization` runs issue #21's charger request and
the independently checked infeasible development case `dev-07` with `tacc-fast`
by default. Use `--profile` to select another profile defined in the checkout's
configuration, such as `cloud-fast` on the Cloud branch.
Pass `--snapshots-root`, `--base-url` (or `--address-file`), `--label` and `--output`.
It records the effective eight-call policy, source hashes, raw model calls,
terminal results and runtime traces. Run the same script in separate source trees
for before/after comparisons. No user answers are invented and waiting runs are
left at `waiting_user`. Checkpoints are isolated in memory for each replay; this
tests decisions, tools and validation rather than Cloud persistence or latency.
Use `--repeats 3 --dev-cases dev-07 dev-04` for review coverage of the infeasible
case and a charger case expected to produce a plan. Development outcomes are also
checked by the independent enumerator/scorer, so clarification on `dev-04` remains
`unnecessary_clarification`. The main historical comparison is in
`docs/experiments/2026-11-17-agent-finalization/REVIEW.md`; the `41750ea` comparison
is supplemental and must not be labeled as the frozen pre-fix baseline.
The main-based port and its separate verification scope are documented in
`docs/experiments/2026-11-17-agent-finalization/MAIN_PORT.md`.
