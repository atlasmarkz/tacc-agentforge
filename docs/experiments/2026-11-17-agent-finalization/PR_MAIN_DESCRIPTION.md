## Main-based Agent finalization fix

This ports only the two Agent fix commits from `fix/cloud-agent-work` onto main:

- `183e631` → `2d3bf7c`
- `1b519a6` → `c7db008`

Base commit: `14a358019aac5240ee1c7c0baf158555aa75ad83`. The branch is named `fix/agent-main-work` as requested. It does not include the Cloud deployment ancestors of PR #27. The root README, knowledge package, deployment files, backend and harness service are unchanged relative to main.

The Agent changes preserve the eight-decision cap, deduplicate facts across reads, reserve a finalization opportunity, keep new recoverable diagnostics available for correction, and expose a scoped `no_feasible_plan` result. That result is covered through the evaluation runner/scorer and the page's English fallback. Clarification on cases that expect a plan remains a scoring failure.

## Main compatibility

Main has no `cloud-fast` profile. The replay CLI now accepts `--profile` and defaults to main's existing `tacc-fast`, recording the actual selected profile in the report. No Cloud configuration is added.

A new entry-point regression runs the default CLI configuration with a scripted model and the real frozen knowledge tools/validator through independent scoring.

## Validation and limits

- Full suite on this main-based tree: **622 passed, 2 skipped**, 14 subtests passed.
- Main replay entry point and page JavaScript syntax/fallback verified.
- No new TACC jobs or real-model replay were run for this main port. The copied raw model records and frozen comparisons are explicitly labeled as historical Cloud-branch evidence, not main measurements or new held-out results.
- No changes to scoring rules or the frozen manifest.

Port details: [MAIN_PORT.md](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/agent-main-work/docs/experiments/2026-11-17-agent-finalization/MAIN_PORT.md).

Related: #21, #27. This PR targets main; #27 retains its Cloud base.
