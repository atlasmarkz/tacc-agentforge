# scripts/

Helper scripts for TACC job submission. Deployment entrypoints currently live in
`deploy/tacc/`; move or wrap them here only if that simplifies team usage.

`python -m scripts.replay_agent_finalization` runs issue #21's charger request and
the independently checked infeasible development case `dev-07` with `cloud-fast`.
Pass `--snapshots-root`, `--base-url` (or `--address-file`), `--label` and `--output`.
It records the effective eight-call policy, source hashes, raw model calls,
terminal results and runtime traces. Run the same script in separate source trees
for before/after comparisons. No user answers are invented and waiting runs are
left at `waiting_user`. Checkpoints are isolated in memory for each replay; this
tests decisions, tools and validation rather than Cloud persistence or latency.
