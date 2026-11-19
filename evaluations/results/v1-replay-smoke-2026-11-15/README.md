# v1 dev replay smoke (2026-11-15)

This is a **harness replay**, not a new model run. The outcomes were recorded by the historical
job 28969 for `v1-dev-01`, `v1-dev-03`, and `v1-dev-04`; they are replayed through the current
data contract, scheduler, scorer, and report code (3 cases x 6 configs = 18 attempts).

Result: A1/B3 3/3, A2 1/3, B0/B1/B2 0/3; BVR denominators exclude unverifiable-cost plans;
`invalid_actions` is 0. These numbers match the historical smoke and validate the current
harness/data/scoring path only. They are not a new model comparison, do not cover the regenerated
dev-00/dev-05 boundary/clarify cases, and must not be used as a model capability result.

The historical source records are `evaluations/results/v1-dev-smoke-28969/records.jsonl`.
