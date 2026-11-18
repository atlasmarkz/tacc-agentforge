## Problem and changes

Issue #21's original charger request exhausted all eight model decisions without delivering a plan or question. The frozen B3v system also hit `model_call_limit` in 48/96 held-out runs; frozen dev-07 used all eight decisions without a useful conclusion.

This branch extends the convergence work already present after #23/#24. It deduplicates candidate/offer/evidence facts rather than whole response pages, marks cached reads as providing no new information, preserves the eight-decision cap and the finalization reserve, and adds a scoped `no_feasible_plan` outcome. Distinct recoverable schema diagnostics retain a correction opportunity; repeated identical diagnostics still count as no progress. Every failed attempt continues to consume the original decision budget.

Clarification is appropriate when missing user requirements change suitability, such as the required laptop wattage and simultaneous charging in the original request. When wattage or device compatibility is already specified and cited product evidence is sufficient, the agent should propose without requesting optional model/brand details. The scorer is unchanged: asking on a case that expects a plan remains `unnecessary_clarification`.

`no_feasible_plan` is included through the shared `agent.nodes.USER_FACING_ISSUES` imported by the evaluation runner. A runtime-to-runner-to-scorer regression verifies that it becomes `abstain`, succeeds on an infeasible case and still fails when a plan was expected. The page's STOP table now provides a catalog-scoped English fallback while preserving model-authored messages.

## Frozen baseline and real replay

The primary pre-fix comparison is the frozen historical system, not the already partially fixed Cloud commit `41750ea`.

| Case | Frozen / original measurement | Current cloud-fast, three repeats |
|---|---|---|
| Original charger request, 60 USD | 12/12 `model_call_limit`, eight-call cap | 1 / 1 / 1 decisions; `waiting_user` × 3 |
| dev-07: keyboard under 10 USD before tax | 8 / 8 / 8; `model_call_limit` × 3 | 3 / 3 / 3; `no_feasible_plan` × 3; independent score 3/3 |
| dev-04: charger ≥30W, 50 USD before tax | 5 / 5 / 5; `completed` × 3 | 5 / 5 / 5; `completed` × 3; independent score 3/3 |

The original charger is not a frozen scored case; its question follows the stated product policy, not a claim that a plan was delivered. If an evaluator defines that request as expecting a plan, the current question must be scored as a failure.

The initial dev-04 review run succeeded only 1/3 times. Its two failures exposed premature finalization after different category/attribute errors; both the failed initial traces and the final 3/3 rerun are retained. This is why the change handles new repair diagnostics separately from repeated ones.

This is a historical frozen-system-to-cumulative-fixes comparison, not an ablation of only this branch's incremental changes. The 96-run held-out set was not rerun and no new aggregate held-out TSR is claimed. The current tests use the same frozen seed with cloud-fast, real tools and budget validation; the temporary TACC model ran in eager mode, unlike the historical compiled service, so latency is not compared. All temporary jobs and replay tunnels have been released.

Full comparison, criteria, source hashes and raw records: [review report](https://github.com/atlasmarkz/tacc-agentforge/blob/fix/cloud-agent-work/docs/experiments/2026-11-17-agent-finalization/REVIEW.md). The earlier Cloud-to-Cloud experiment remains explicitly labeled as supplemental.

## Validation

- `agent/.venv/bin/python -m pytest agent/tests tests -q -o faulthandler_timeout=40`: **653 passed, 5 skipped**, 16 subtests passed.
- Nine final real-model runs; call counts equal recorded `generate()` invocations and never exceed eight. Both development cases pass the existing independent enumerator/scorer in all three final repeats.
- Page JavaScript syntax and the new English fallback checked with Node.
- No changes to the frozen manifest, sealed held-out cases or scoring rules.

Related: #21.
